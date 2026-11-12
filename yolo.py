import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from evaluar import evaluar, guardar, resumen

RAIZ = Path(__file__).resolve().parent
COPIA = RAIZ.parent / "dataset" / "FjordFish_T2"
ENTRADA = 640
SEMILLA = 42


def nombre_modelo(args):
    return f"yolo_{args.pesos}" + ("_congelado" if args.congelar else "")


def entrenar(args):
    base = RAIZ / "resultados" / nombre_modelo(args)
    t0 = time.time()
    if args.reanudar:
        YOLO(base / "entrenamiento" / "weights" / "last.pt").train(resume=True)
    else:
        if args.pesos == "cero":
            modelo = YOLO(f"yolo11{args.tamano}.yaml")
            assert not modelo.ckpt and modelo.ckpt_path is None, "el modelo desde cero no debe traer pesos"
        else:
            modelo = YOLO(f"yolo11{args.tamano}.pt")
        modelo.train(
            data=str(COPIA / "fjordfish.yaml"), epochs=args.epocas, imgsz=ENTRADA,
            batch=args.batch, workers=args.workers, seed=SEMILLA, deterministic=True,
            pretrained=args.pesos == "coco", freeze=args.congelar or None,
            optimizer="SGD", lr0=args.lr, momentum=0.937, weight_decay=5e-4, warmup_epochs=3, cos_lr=True,
            patience=args.paciencia,
            mosaic=1.0, close_mosaic=10, scale=0.5, translate=0.1, fliplr=0.5, flipud=0.0,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            project=str(base), name="entrenamiento", exist_ok=True, plots=True, verbose=False)
    print(f"Entrenamiento terminado en {(time.time() - t0) / 60:.1f} min")


def evaluar_modelo(args):
    base = RAIZ / "resultados" / nombre_modelo(args)
    modelo = YOLO(base / "entrenamiento" / "weights" / "best.pt")
    parametros = sum(p.numel() for p in modelo.model.parameters())
    congelados = sum(p.numel() for capa in modelo.model.model[:args.congelar] for p in capa.parameters())
    coco = json.loads((COPIA / "coco" / f"{args.split}.json").read_text(encoding="utf-8"))

    # una imagen por llamada: Ultralytics procesa una lista de rutas como un solo lote
    preds = []
    for im in coco["images"]:
        r = modelo.predict(str(COPIA / im["file_name"]), imgsz=ENTRADA, conf=0.001, iou=0.7, max_det=100,
                           quantize=16, verbose=False)[0]
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            preds.append({"image_id": im["id"], "category_id": int(c) + 1, "score": round(s, 5),
                          "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]})
    (base / f"predicciones_{args.split}.json").write_text(json.dumps(preds), encoding="utf-8")
    res = evaluar(preds, args.split)

    extra = {"modelo": f"yolo11{args.tamano}_{args.pesos}" + ("_congelado" if args.congelar else ""),
             "parametros": parametros, "capas_congeladas": args.congelar,
             "parametros_congelados": congelados, "parametros_entrenables": parametros - congelados,
             "entrada": ENTRADA}
    with open(base / "entrenamiento" / "results.csv", encoding="utf-8") as f:
        hist = [{k.strip(): v for k, v in fila.items()} for fila in csv.DictReader(f)]
    mapa = [float(h["metrics/mAP50-95(B)"]) for h in hist]
    extra.update(epocas=len(hist), mejor_epoca=int(np.argmax(mapa)) + 1,
                 val_mAP50_95_ultralytics=round(max(mapa), 4),
                 tiempo_entrenamiento_min=round(float(hist[-1]["time"]) / 60, 1))
    guardar(res, base, extra)
    print(resumen(res, f"{extra['modelo']} ({args.split})"))
    print(f"Parámetros: {parametros / 1e6:.2f} M  |  guardado en {base}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("modo", choices=["entrenar", "evaluar"])
    ap.add_argument("--pesos", choices=["cero", "coco"], required=True)
    ap.add_argument("--tamano", default="s", choices=["n", "s", "m"])
    ap.add_argument("--epocas", type=int, default=160)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--paciencia", type=int, default=50, help="épocas sin mejorar en val antes de detener")
    ap.add_argument("--congelar", type=int, default=0, help="capas iniciales congeladas (11 = todo el backbone)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--reanudar", action="store_true", help="continuar desde last.pt")
    args = ap.parse_args()
    if args.modo == "entrenar":
        entrenar(args)
    evaluar_modelo(args)


if __name__ == "__main__":
    main()
