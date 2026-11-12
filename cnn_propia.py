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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from evaluar import cargar_gt, evaluar, guardar, resumen

cv2.setNumThreads(0)

RAIZ = Path(__file__).resolve().parent
COPIA = RAIZ.parent / "dataset" / "FjordFish_T2"
SALIDA = RAIZ / "resultados" / "cnn_propia"
N_CLASES = 28
ENTRADA = 640
STRIDE = 4
MAX_OBJ = 128
SEMILLA = 42


# ---------------------------------------------------------------- datos
def jitter_hsv(img, h=0.015, s=0.7, v=0.4):
    """Variación aleatoria de tono, saturación y brillo."""
    r = [random.uniform(-1, 1) * k + 1 for k in (h, s, v)]
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    luts = [((x * r[0]) % 180), np.clip(x * r[1], 0, 255), np.clip(x * r[2], 0, 255)]
    hsv = cv2.merge([cv2.LUT(c, l.astype(np.uint8)) for c, l in zip((hue, sat, val), luts)])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def radio_gaussiano(h, w, min_overlap=0.7):
    """Radio dentro del cual un centro desplazado mantiene IoU >= min_overlap (CornerNet)."""
    b1, c1 = h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    b2, c2 = 2 * (h + w), (1 - min_overlap) * w * h
    b3, c3 = -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * c1)) / 2
    r2 = (b2 + math.sqrt(b2 ** 2 - 16 * c2)) / 2
    r3 = (b3 + math.sqrt(b3 ** 2 - 16 * min_overlap * c3)) / 2
    return min(r1, r2, r3)


def dibujar_gaussiana(hm, cx, cy, r):
    d = 2 * r + 1
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(x * x + y * y) / (2 * (d / 6) ** 2))
    H, W = hm.shape
    izq, der, arr, aba = min(cx, r), min(W - cx, r + 1), min(cy, r), min(H - cy, r + 1)
    zona = hm[cy - arr:cy + aba, cx - izq:cx + der]
    np.maximum(zona, g[r - arr:r + aba, r - izq:r + der], out=zona)


def codificar(cajas):
    """Cajas (x1, y1, x2, y2, clase) en píxeles de la entrada -> objetivos de CenterNet."""
    S = ENTRADA // STRIDE
    hm = np.zeros((N_CLASES, S, S), np.float32)
    wh, reg = np.zeros((MAX_OBJ, 2), np.float32), np.zeros((MAX_OBJ, 2), np.float32)
    ind, mask = np.zeros(MAX_OBJ, np.int64), np.zeros(MAX_OBJ, np.float32)
    for k, (x1, y1, x2, y2, c) in enumerate(cajas[:MAX_OBJ]):
        w, h = (x2 - x1) / STRIDE, (y2 - y1) / STRIDE
        cx, cy = (x1 + x2) / 2 / STRIDE, (y1 + y2) / 2 / STRIDE
        ci, cj = min(int(cx), S - 1), min(int(cy), S - 1)
        dibujar_gaussiana(hm[int(c)], ci, cj, max(0, int(radio_gaussiano(math.ceil(h), math.ceil(w)))))
        wh[k], reg[k] = (w, h), (cx - ci, cy - cj)
        ind[k], mask[k] = cj * S + ci, 1
    return hm, wh, reg, ind, mask


class FjordFish(Dataset):
    
    def __init__(self, split, aumentar=False):
        coco = json.loads((COPIA / "coco" / f"{split}.json").read_text(encoding="utf-8"))
        self.imagenes = coco["images"]
        cajas = {im["id"]: [] for im in self.imagenes}
        for a in coco["annotations"]:
            cajas[a["image_id"]].append([*a["bbox"], a["category_id"] - 1])
        self.cajas = {k: np.array(v, np.float32).reshape(-1, 5) for k, v in cajas.items()}
        self.aumentar = aumentar

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, i):
        info = self.imagenes[i]
        img = cv2.imread(str(COPIA / info["file_name"]))
        h0, w0 = img.shape[:2]
        escala = ENTRADA / max(h0, w0)
        s, tx, ty, voltear = escala, 0.0, 0.0, False
        if self.aumentar:
            s = escala * random.uniform(0.6, 1.4)
            nw, nh = w0 * s, h0 * s
            tx = random.uniform(min(0, ENTRADA - nw), max(0, ENTRADA - nw))
            ty = random.uniform(min(0, ENTRADA - nh), max(0, ENTRADA - nh))
            voltear = random.random() < 0.5
        M = np.array([[-s if voltear else s, 0, ENTRADA - tx if voltear else tx], [0, s, ty]], np.float32)
        img = cv2.warpAffine(img, M, (ENTRADA, ENTRADA), flags=cv2.INTER_LINEAR,
                             borderValue=(114, 114, 114))
        if self.aumentar:
            img = jitter_hsv(img)

        c = self.cajas[info["id"]]
        x1, y1 = c[:, 0] * s + tx, c[:, 1] * s + ty
        x2, y2 = (c[:, 0] + c[:, 2]) * s + tx, (c[:, 1] + c[:, 3]) * s + ty
        if voltear:
            x1, x2 = ENTRADA - x2, ENTRADA - x1
        area0 = (x2 - x1) * (y2 - y1)
        x1, x2, y1, y2 = (np.clip(v, 0, ENTRADA) for v in (x1, x2, y1, y2))
        # se descartan las cajas que quedan casi fuera de la imagen
        quedan = ((x2 - x1) * (y2 - y1) > 0.3 * area0) & (x2 - x1 > 2) & (y2 - y1 > 2)
        hm, wh, reg, ind, mask = codificar(np.stack([x1, y1, x2, y2, c[:, 4]], 1)[quedan])
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))
        return (x, torch.from_numpy(hm), torch.from_numpy(wh), torch.from_numpy(reg),
                torch.from_numpy(ind), torch.from_numpy(mask), info["id"], escala, w0, h0)


def normalizar(x):
    return (x.float() / 255 - 0.5) / 0.25


# ---------------------------------------------------------------- modelo
def conv_bn(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class BloqueResidual(nn.Module):

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout)
        self.c2, self.b2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout)
        nn.init.zeros_(self.b2.weight)
        self.atajo = (nn.Identity() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        return F.relu(self.b2(self.c2(y)) + self.atajo(x), inplace=True)


def cabeza(cin, salidas):
    return nn.Sequential(nn.Conv2d(cin, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(64, salidas, 1))


class CNNPropia(nn.Module):
    def __init__(self, canales=(64, 128, 256, 512), bloques=(2, 2, 2, 2), neck=128):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(3, canales[0], 7, 2), nn.MaxPool2d(3, 2, 1))
        etapas, cin = [], canales[0]
        for i, (c, n) in enumerate(zip(canales, bloques)):
            etapas.append(nn.Sequential(*[BloqueResidual(cin if j == 0 else c, c, 2 if (j == 0 and i > 0) else 1)
                                          for j in range(n)]))
            cin = c
        self.etapas = nn.ModuleList(etapas)
        self.laterales = nn.ModuleList(nn.Conv2d(c, neck, 1) for c in canales)
        self.suavizado = nn.ModuleList(conv_bn(neck, neck) for _ in canales[:-1])
        self.hm, self.wh, self.reg = cabeza(neck, N_CLASES), cabeza(neck, 2), cabeza(neck, 2)
        # sesgo inicial del mapa de calor: probabilidad 0.1, estabiliza la pérdida focal
        nn.init.constant_(self.hm[-1].bias, -2.19)

    def forward(self, x):
        feats, y = [], self.stem(x)
        for etapa in self.etapas:
            y = etapa(y)
            feats.append(y)
        p = self.laterales[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            p = F.interpolate(p, size=feats[i].shape[-2:], mode="nearest") + self.laterales[i](feats[i])
            p = self.suavizado[i](p)
        return self.hm(p), self.wh(p), self.reg(p)


# ---------------------------------------------------------------- pérdida y decodificación
def perdida_focal(logits, gt):
    p = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    l_pos = torch.log(p) * (1 - p) ** 2 * pos
    l_neg = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * (1 - pos)
    return -(l_pos.sum() + l_neg.sum()) / pos.sum().clamp(min=1)


def recoger(mapa, ind):
    """Valores de un mapa (B, C, H, W) en las posiciones ind (B, K) -> (B, K, C)."""
    B, C = mapa.shape[:2]
    return mapa.reshape(B, C, -1).gather(2, ind.unsqueeze(1).expand(B, C, ind.shape[1])).permute(0, 2, 1)


def perdida(salida, hm, wh, reg, ind, mask):
    l_hm = perdida_focal(salida[0], hm)
    m = mask.unsqueeze(2).expand(-1, -1, 2)
    l_wh = (recoger(salida[1].float(), ind) - wh).abs().mul(m).sum() / (m.sum() + 1e-4)
    l_reg = (recoger(salida[2].float(), ind) - reg).abs().mul(m).sum() / (m.sum() + 1e-4)
    return l_hm + 0.1 * l_wh + l_reg, (l_hm.item(), l_wh.item(), l_reg.item())


@torch.no_grad()
def decodificar(salida, k=100):
    # Máximos del mapa de calor -> (cajas x1y1x2y2 en píxeles de la entrada, puntajes, clases).
    hm = salida[0].float().sigmoid()
    hm = hm * (F.max_pool2d(hm, 3, 1, 1) == hm)
    B, C, H, W = hm.shape
    puntajes, idx = hm.reshape(B, -1).topk(k)
    clases, pix = idx // (H * W), idx % (H * W)
    ys, xs = (pix // W).float(), (pix % W).float()
    wh, reg = recoger(salida[1].float(), pix), recoger(salida[2].float(), pix)
    cx, cy, w, h = xs + reg[..., 0], ys + reg[..., 1], wh[..., 0], wh[..., 1]
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) * STRIDE, puntajes, clases


def a_coco(cajas, puntajes, clases, ids, escalas, anchos, altos, umbral=0.01):
    # Detecciones de un lote -> predicciones COCO en píxeles de la copia reducida.
    preds = []
    for b in range(len(ids)):
        c = (cajas[b] / float(escalas[b])).cpu().numpy()
        c[:, [0, 2]] = c[:, [0, 2]].clip(0, float(anchos[b]))
        c[:, [1, 3]] = c[:, [1, 3]].clip(0, float(altos[b]))
        for (x1, y1, x2, y2), s, k in zip(c, puntajes[b].tolist(), clases[b].tolist()):
            if s >= umbral and x2 > x1 and y2 > y1:
                preds.append({"image_id": int(ids[b]), "category_id": int(k) + 1, "score": round(s, 5),
                              "bbox": [round(float(x1), 2), round(float(y1), 2),
                                       round(float(x2 - x1), 2), round(float(y2 - y1), 2)]})
    return preds


def cargador_eval(ds, batch=16, workers=4, persistente=False):
    return DataLoader(ds, batch, num_workers=workers, pin_memory=True,
                      persistent_workers=persistente and workers > 0)


@torch.no_grad()
def predecir(modelo, cargador):
    modelo.eval()
    dev = next(modelo.parameters()).device
    preds = []
    for x, *_, ids, esc, an, al in cargador:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            salida = modelo(normalizar(x.to(dev, non_blocking=True)))
        preds += a_coco(*decodificar(salida), ids, esc, an, al)
    return preds


# ---------------------------------------------------------------- entrenamiento
def crear_optimizador(modelo, lr, wd):
    # AdamW, sin weight decay en BatchNorm ni sesgos.
    con, sin = [], []
    for p in modelo.parameters():
        (con if p.ndim > 1 else sin).append(p)
    return torch.optim.AdamW([{"params": con, "weight_decay": wd}, {"params": sin, "weight_decay": 0}], lr=lr)


def paso(modelo, opt, escalador, lote, dev):
    x, hm, wh, reg, ind, mask = (t.to(dev, non_blocking=True) for t in lote[:6])
    with torch.autocast("cuda", dtype=torch.float16):
        salida = modelo(normalizar(x))
    l, partes = perdida(salida, hm, wh, reg, ind, mask)
    opt.zero_grad(set_to_none=True)
    escalador.scale(l).backward()
    escalador.unscale_(opt)
    nn.utils.clip_grad_norm_(modelo.parameters(), 10.0)
    escalador.step(opt)
    escalador.update()
    return l.item(), partes


def entrenar(args):
    dev = torch.device("cuda")
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    torch.backends.cudnn.benchmark = True
    SALIDA.mkdir(parents=True, exist_ok=True)

    dl_val = cargador_eval(FjordFish("val"), args.batch, min(4, args.workers), persistente=True)
    dl = DataLoader(FjordFish("train", aumentar=True), args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    modelo = CNNPropia().to(dev)
    opt = crear_optimizador(modelo, args.lr, args.wd)
    # calentamiento lineal de 2 épocas y decaimiento coseno hasta el 1 % del lr
    total, calent = args.epocas * len(dl), 2 * len(dl)
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

    print(f"Entrenando {sum(p.numel() for p in modelo.parameters()) / 1e6:.2f} M parámetros, "
          f"{len(dl)} lotes de {args.batch} por época, hasta {args.epocas} épocas")
    for ep in range(estado["epoca"], args.epocas):
        modelo.train()
        t0, suma, n = time.time(), np.zeros(4), 0
        for lote in dl:
            l, partes = paso(modelo, opt, escalador, lote, dev)
            sched.step()
            suma += (l, *partes)
            n += 1
        t_ep = time.time() - t0
        m = evaluar(predecir(modelo, dl_val), "val", gt_val)
        estado["segundos"] += time.time() - t0
        fila = {"epoca": ep + 1, "perdida": suma[0] / n, "p_heatmap": suma[1] / n, "p_tamano": suma[2] / n,
                "p_desfase": suma[3] / n, "val_mAP50_95": m["mAP50_95"], "val_mAP50": m["mAP50"],
                "val_precision": m["precision"], "val_recall": m["recall"],
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
        print(f"época {ep + 1:3d}/{args.epocas}  pérdida {fila['perdida']:.3f} (hm {fila['p_heatmap']:.3f} "
              f"tam {fila['p_tamano']:.2f} des {fila['p_desfase']:.3f})  val mAP50-95 {m['mAP50_95']:.4f} "
              f"mAP50 {m['mAP50']:.4f}  {t_ep:.0f} s{'  *mejor*' if mejoro else ''}", flush=True)
        if estado["sin_mejora"] >= args.paciencia:
            print(f"Early stopping: {args.paciencia} épocas sin mejorar en val")
            break
    print(f"Entrenamiento terminado en {estado['segundos'] / 60:.1f} min. Mejor época: "
          f"{estado['mejor_epoca']} (val mAP50-95 {estado['mejor']:.4f})")
    graficar(estado["historial"])
    evaluar_modelo(argparse.Namespace(split="test", batch=args.batch, workers=args.workers))


def graficar(historial):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoca"] for h in historial]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, et in (("perdida", "total"), ("p_heatmap", "mapa de calor"), ("p_desfase", "desfase")):
        ax[0].plot(ep, [h[k] for h in historial], label=et)
    ax[0].set(xlabel="Época", ylabel="Pérdida (train)", title="Pérdida de entrenamiento", yscale="log")
    ax[0].legend(frameon=False)
    for k, et in (("val_mAP50", "mAP50"), ("val_mAP50_95", "mAP50-95")):
        ax[1].plot(ep, [h[k] for h in historial], label=et)
    ax[1].set(xlabel="Época", ylabel="val", title="mAP en validación")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SALIDA / "curvas.png", dpi=150)
    plt.close(fig)


def evaluar_modelo(args):
    dev = torch.device("cuda")
    modelo = CNNPropia().to(dev)
    modelo.load_state_dict(torch.load(SALIDA / "mejor.pt", map_location=dev))
    preds = predecir(modelo, cargador_eval(FjordFish(args.split), args.batch, args.workers))
    (SALIDA / f"predicciones_{args.split}.json").write_text(json.dumps(preds), encoding="utf-8")
    res = evaluar(preds, args.split)
    extra = {"modelo": "cnn_propia", "parametros": sum(p.numel() for p in modelo.parameters()),
             "entrada": ENTRADA}
    if (SALIDA / "ultimo.pt").exists():
        est = torch.load(SALIDA / "ultimo.pt", map_location="cpu", weights_only=False)["estado"]
        extra.update(tiempo_entrenamiento_min=round(est["segundos"] / 60, 1), epocas=est["epoca"],
                     mejor_epoca=est["mejor_epoca"])
    guardar(res, SALIDA, extra)
    print(resumen(res, f"cnn_propia ({args.split})"))
    print(f"Parámetros: {extra['parametros'] / 1e6:.2f} M  |  guardado en {SALIDA}")


def visualizar_aumentos(n=6):
    """Guarda aumentos.png: imágenes de train aumentadas con sus cajas y su mapa de calor objetivo."""
    random.seed(SEMILLA)
    SALIDA.mkdir(parents=True, exist_ok=True)
    ds, S, filas = FjordFish("train", aumentar=True), ENTRADA // STRIDE, []
    for i in random.sample(range(len(ds)), n):
        x, hm, wh, reg, ind, mask, *_ = ds[i]
        img = np.ascontiguousarray(x.numpy().transpose(1, 2, 0)[:, :, ::-1])
        for k in range(int(mask.sum())):
            cx, cy = ind[k] % S + reg[k, 0], ind[k] // S + reg[k, 1]
            w, h = wh[k]
            cv2.rectangle(img, (int((cx - w / 2) * STRIDE), int((cy - h / 2) * STRIDE)),
                          (int((cx + w / 2) * STRIDE), int((cy + h / 2) * STRIDE)), (0, 255, 255), 2)
        calor = cv2.applyColorMap((hm.max(0).values.numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
        filas.append(np.hstack([img, cv2.resize(calor, (ENTRADA, ENTRADA), interpolation=cv2.INTER_NEAREST)]))
    cv2.imwrite(str(SALIDA / "aumentos.png"),
                cv2.resize(np.vstack(filas), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("modo", choices=["entrenar", "evaluar", "aumentos"])
    ap.add_argument("--epocas", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    ap.add_argument("--paciencia", type=int, default=25, help="épocas sin mejorar en val antes de detener")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--reanudar", action="store_true", help="continuar desde ultimo.pt")
    args = ap.parse_args()
    {"entrenar": entrenar, "evaluar": evaluar_modelo, "aumentos": lambda _: visualizar_aumentos()}[args.modo](args)


if __name__ == "__main__":
    main()
