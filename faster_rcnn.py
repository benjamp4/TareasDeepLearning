import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from cnn_propia import jitter_hsv
from evaluar import cargar_gt, evaluar, guardar, resumen

cv2.setNumThreads(0)

RAIZ = Path(__file__).resolve().parent
COPIA = RAIZ.parent / "dataset" / "FjordFish_T2"
SALIDA = RAIZ / "resultados" / "faster_rcnn"
N_CLASES = 28
ENTRADA = 640
SEMILLA = 42
BLOQUES = {"backbone.body.conv1": "stem (conv1)", "backbone.body.layer1": "ResNet layer1",
           "backbone.body.layer2": "ResNet layer2", "backbone.body.layer3": "ResNet layer3",
           "backbone.body.layer4": "ResNet layer4", "backbone.fpn": "FPN", "rpn": "RPN",
           "roi_heads.box_head": "cabezal ROI", "roi_heads.box_predictor": "capa final (nueva)"}


# ---------------------------------------------------------------- datos
class FjordFishRCNN(Dataset):
    """Imagen (tensor 0-1) y cajas x1y1x2y2 en píxeles de la copia reducida.

    El modelo redimensiona a ENTRADA internamente. Aumentos (train): volteo
    horizontal y color HSV, los mismos de la CNN propia.
    """

    def __init__(self, split, aumentar=False):
        coco = json.loads((COPIA / "coco" / f"{split}.json").read_text(encoding="utf-8"))
        self.imagenes = coco["images"]
        cajas = {im["id"]: [] for im in self.imagenes}
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            cajas[a["image_id"]].append([x, y, x + w, y + h, a["category_id"]])
        self.cajas = {k: np.array(v, np.float32).reshape(-1, 5) for k, v in cajas.items()}
        self.aumentar = aumentar

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, i):
        info = self.imagenes[i]
        img = cv2.imread(str(COPIA / info["file_name"]))
        c = self.cajas[info["id"]].copy()
        if self.aumentar:
            if random.random() < 0.5:
                img = img[:, ::-1]
                c[:, [0, 2]] = img.shape[1] - c[:, [2, 0]]
            img = jitter_hsv(np.ascontiguousarray(img))
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))).float() / 255
        objetivo = {"boxes": torch.from_numpy(c[:, :4]), "labels": torch.from_numpy(c[:, 4]).long()}
        return x, objetivo, info["id"]


def juntar(lote):
    return tuple(zip(*lote))


def cargador(split, batch, workers, aumentar=False, persistente=False):
    return DataLoader(FjordFishRCNN(split, aumentar), batch, shuffle=aumentar, num_workers=workers,
                      collate_fn=juntar, pin_memory=True, drop_last=aumentar,
                      persistent_workers=persistente and workers > 0)


# ---------------------------------------------------------------- modelo
def crear_modelo(capas_entrenables=3):
    modelo = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
                                     trainable_backbone_layers=capas_entrenables,
                                     min_size=ENTRADA, max_size=ENTRADA)
    entrada = modelo.roi_heads.box_predictor.cls_score.in_features
    modelo.roi_heads.box_predictor = FastRCNNPredictor(entrada, N_CLASES + 1)
    modelo.roi_heads.detections_per_img = 100
    return modelo


def resumen_congelamiento(modelo):
    """Parámetros totales y entrenables de cada bloque del modelo."""
    filas = []
    for prefijo, nombre in BLOQUES.items():
        ps = [p for n, p in modelo.named_parameters() if n.startswith(prefijo)]
        entren = sum(p.numel() for p in ps if p.requires_grad)
        total = sum(p.numel() for p in ps)
        estado = "entrena" if entren == total else ("congelado" if entren == 0 else "parcial")
        filas.append({"bloque": nombre, "parametros": total, "entrenables": entren, "estado": estado})
    return filas


@torch.no_grad()
def predecir(modelo, dl):
    modelo.eval()
    modelo.roi_heads.score_thresh = 0.001
    dev = next(modelo.parameters()).device
    preds = []
    for imgs, _, ids in dl:
        imgs = [i.to(dev, non_blocking=True) for i in imgs]
        with torch.autocast("cuda", dtype=torch.float16):
            salidas = modelo(imgs)
        for o, iid in zip(salidas, ids):
            for (x1, y1, x2, y2), s, c in zip(o["boxes"].float().tolist(), o["scores"].float().tolist(),
                                              o["labels"].tolist()):
                preds.append({"image_id": int(iid), "category_id": int(c), "score": round(s, 5),
                              "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]})
    return preds


# ---------------------------------------------------------------- entrenamiento
def paso(modelo, opt, escalador, imgs, objetivos, dev):
    imgs = [i.to(dev, non_blocking=True) for i in imgs]
    objetivos = [{k: v.to(dev, non_blocking=True) for k, v in t.items()} for t in objetivos]
    with torch.autocast("cuda", dtype=torch.float16):
        perdidas = modelo(imgs, objetivos)
    total = sum(perdidas.values())
    opt.zero_grad(set_to_none=True)
    escalador.scale(total).backward()
    escalador.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], 10.0)
    escalador.step(opt)
    escalador.update()
    return total.item(), {k: v.item() for k, v in perdidas.items()}


def entrenar(args):
    dev = torch.device("cuda")
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    SALIDA.mkdir(parents=True, exist_ok=True)
    dl = cargador("train", args.batch, args.workers, aumentar=True, persistente=True)
    dl_val = cargador("val", args.batch, min(4, args.workers), persistente=True)
    modelo = crear_modelo(args.capas_entrenables).to(dev)
    entrenables = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.SGD(entrenables, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # calentamiento lineal de 1 época y decaimiento coseno hasta el 1 % del lr
    total, calent = args.epocas * len(dl), len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / calent) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(it, total) / total))))
    escalador = torch.amp.GradScaler("cuda")
    gt_val = cargar_gt("val")
    estado = {"epoca": 0, "mejor": -1.0, "mejor_epoca": -1, "sin_mejora": 0, "historial": [], "segundos": 0.0}
    if args.reanudar and (SALIDA / "ultimo.pt").exists():
        ck = torch.load(SALIDA / "ultimo.pt", map_location=dev, weights_only=False)
        for obj, clave in ((modelo, "modelo"), (opt, "opt"), (sched, "sched"), (escalador, "escalador")):
            obj.load_state_dict(ck[clave])
        estado = ck["estado"]
        print(f"Reanudando desde la época {estado['epoca']}")

    n_ent = sum(p.numel() for p in entrenables)
    n_tot = sum(p.numel() for p in modelo.parameters())
    print(f"Faster R-CNN: {n_tot / 1e6:.1f} M parámetros, {n_ent / 1e6:.1f} M entrenables; "
          f"{len(dl)} lotes de {args.batch} por época, hasta {args.epocas} épocas")
    for ep in range(estado["epoca"], args.epocas):
        modelo.train()
        t0, suma, n = time.time(), {}, 0
        for imgs, objetivos, _ in dl:
            l, partes = paso(modelo, opt, escalador, imgs, objetivos, dev)
            sched.step()
            for k, v in {"perdida": l, **partes}.items():
                suma[k] = suma.get(k, 0) + v
            n += 1
        t_ep = time.time() - t0
        m = evaluar(predecir(modelo, dl_val), "val", gt_val)
        estado["segundos"] += time.time() - t0
        fila = {"epoca": ep + 1, **{k: v / n for k, v in suma.items()}, "val_mAP50_95": m["mAP50_95"],
                "val_mAP50": m["mAP50"], "val_precision": m["precision"], "val_recall": m["recall"],
                "lr": opt.param_groups[0]["lr"], "seg_epoca": t_ep}
        estado["historial"].append(fila)
        mejoro = m["mAP50_95"] > estado["mejor"]
        if mejoro:
            estado.update(mejor=m["mAP50_95"], mejor_epoca=ep + 1, sin_mejora=0)
            torch.save(modelo.state_dict(), SALIDA / "mejor.pt")
        else:
            estado["sin_mejora"] += 1
        estado["epoca"] = ep + 1
        torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "escalador": escalador.state_dict(), "estado": estado}, SALIDA / "ultimo.pt")
        with open(SALIDA / "historial.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(fila))
            wr.writeheader()
            wr.writerows(estado["historial"])
        print(f"época {ep + 1:3d}/{args.epocas}  pérdida {fila['perdida']:.3f}  val mAP50-95 {m['mAP50_95']:.4f} "
              f"mAP50 {m['mAP50']:.4f}  {t_ep:.0f} s{'  *mejor*' if mejoro else ''}", flush=True)
        if estado["sin_mejora"] >= args.paciencia:
            print(f"Early stopping: {args.paciencia} épocas sin mejorar en val")
            break
    print(f"Entrenamiento terminado en {estado['segundos'] / 60:.1f} min. Mejor época: "
          f"{estado['mejor_epoca']} (val mAP50-95 {estado['mejor']:.4f})")
    graficar(estado["historial"])
    evaluar_modelo(argparse.Namespace(split="test", batch=args.batch, workers=args.workers,
                                      capas_entrenables=args.capas_entrenables))


def graficar(historial):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoca"] for h in historial]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k in ("perdida", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
        if k in historial[0]:
            ax[0].plot(ep, [h[k] for h in historial], label=k)
    ax[0].set(xlabel="Época", ylabel="Pérdida (train)", title="Pérdida de entrenamiento", yscale="log")
    ax[0].legend(frameon=False, fontsize=8)
    for k, et in (("val_mAP50", "mAP50"), ("val_mAP50_95", "mAP50-95")):
        ax[1].plot(ep, [h[k] for h in historial], label=et)
    ax[1].set(xlabel="Época", ylabel="val", title="mAP en validación")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SALIDA / "curvas.png", dpi=150)
    plt.close(fig)


def evaluar_modelo(args):
    dev = torch.device("cuda")
    modelo = crear_modelo(args.capas_entrenables).to(dev)
    modelo.load_state_dict(torch.load(SALIDA / "mejor.pt", map_location=dev))
    preds = predecir(modelo, cargador(args.split, args.batch, args.workers))
    (SALIDA / f"predicciones_{args.split}.json").write_text(json.dumps(preds), encoding="utf-8")
    res = evaluar(preds, args.split)
    congel = resumen_congelamiento(modelo)
    extra = {"modelo": "faster_rcnn_resnet50_fpn_coco",
             "parametros": sum(p.numel() for p in modelo.parameters()),
             "parametros_entrenables": sum(f["entrenables"] for f in congel),
             "entrada": ENTRADA, "congelamiento": congel}
    if (SALIDA / "ultimo.pt").exists():
        est = torch.load(SALIDA / "ultimo.pt", map_location="cpu", weights_only=False)["estado"]
        extra.update(tiempo_entrenamiento_min=round(est["segundos"] / 60, 1), epocas=est["epoca"],
                     mejor_epoca=est["mejor_epoca"])
    guardar(res, SALIDA, extra)
    print(resumen(res, f"faster_rcnn ({args.split})"))
    print(f"Parámetros: {extra['parametros'] / 1e6:.1f} M ({extra['parametros_entrenables'] / 1e6:.1f} M "
          f"entrenables)  |  guardado en {SALIDA}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("modo", choices=["entrenar", "evaluar"])
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--capas-entrenables", type=int, default=3, choices=range(6),
                    help="etapas del ResNet-50 que se entrenan (0 = backbone congelado, 5 = todo)")
    ap.add_argument("--paciencia", type=int, default=8, help="épocas sin mejorar en val antes de detener")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--reanudar", action="store_true", help="continuar desde ultimo.pt")
    args = ap.parse_args()
    entrenar(args) if args.modo == "entrenar" else evaluar_modelo(args)


if __name__ == "__main__":
    main()
