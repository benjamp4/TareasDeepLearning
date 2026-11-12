import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

RAIZ = Path(__file__).resolve().parent
COPIA = RAIZ.parent / "dataset" / "FjordFish_T2"
RES = RAIZ / "resultados"
N_IMAGENES = 300


def cargar_imagenes():
    coco = json.loads((COPIA / "coco" / "test.json").read_text(encoding="utf-8"))
    ims = sorted(coco["images"], key=lambda im: im["id"])[:N_IMAGENES]
    return [cv2.imread(str(COPIA / im["file_name"])) for im in ims]


def cronometrar(predecir, imagenes):
    """Milisegundos por imagen en la segunda pasada; la primera es de calentamiento."""
    ms = []
    with torch.no_grad():
        for img in imagenes:
            predecir(img)
        for img in imagenes:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            predecir(img)
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
    return np.array(ms)


def cnn_propia(dev):
    from cnn_propia import CNNPropia, ENTRADA, decodificar, normalizar
    modelo = CNNPropia().to(dev).eval()
    modelo.load_state_dict(torch.load(RES / "cnn_propia" / "mejor.pt", map_location=dev))

    def predecir(img):
        h0, w0 = img.shape[:2]
        s = ENTRADA / max(h0, w0)
        img = cv2.warpAffine(img, np.float32([[s, 0, 0], [0, s, 0]]), (ENTRADA, ENTRADA),
                             flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))[None].to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            cajas, puntajes, clases = decodificar(modelo(normalizar(x)))
        return (cajas[0] / s).cpu(), puntajes[0].cpu(), clases[0].cpu()

    return modelo, predecir


def yolo(carpeta):
    def crear(dev):
        from ultralytics import YOLO
        modelo = YOLO(RES / carpeta / "entrenamiento" / "weights" / "best.pt")

        def predecir(img):
            r = modelo.predict(img, imgsz=640, conf=0.001, iou=0.7, max_det=100, quantize=16,
                               device=0, verbose=False)[0]
            return r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu()

        return modelo.model, predecir
    return crear


def faster_rcnn(dev):
    from faster_rcnn import crear_modelo
    modelo = crear_modelo().to(dev).eval()
    modelo.load_state_dict(torch.load(RES / "faster_rcnn" / "mejor.pt", map_location=dev))
    modelo.roi_heads.score_thresh = 0.001

    def predecir(img):
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))).to(dev).float() / 255
        with torch.autocast("cuda", dtype=torch.float16):
            o = modelo([x])[0]
        return o["boxes"].cpu(), o["scores"].cpu(), o["labels"].cpu()

    return modelo, predecir


# nombre -> (carpeta en resultados/, función que crea el modelo)
MODELOS = {
    "CNN propia (cero)": ("cnn_propia", cnn_propia),
    "YOLO11s (cero)": ("yolo_cero", yolo("yolo_cero")),
    "Faster R-CNN (COCO)": ("faster_rcnn", faster_rcnn),
    "YOLO11s (COCO, congelado)": ("yolo_coco_congelado", yolo("yolo_coco_congelado")),
    "YOLO11s (COCO, completo)": ("yolo_coco", yolo("yolo_coco")),
}


def actualizar_metricas():
    """Copia los FPS de fps.csv a cada metricas_test.json."""
    with open(RES / "comparacion" / "fps.csv", encoding="utf-8") as f:
        fps = {fila["modelo"]: fila for fila in csv.DictReader(f)}
    for nombre, (carpeta, _) in MODELOS.items():
        ruta = RES / carpeta / "metricas_test.json"
        m = json.loads(ruta.read_text(encoding="utf-8"))
        m["fps"] = float(fps[nombre]["fps"])
        m["ms_por_imagen"] = float(fps[nombre]["ms_media"])
        ruta.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    dev = torch.device("cuda")
    imagenes = cargar_imagenes()
    print(f"{len(imagenes)} imágenes, lote 1, fp16, {torch.cuda.get_device_name(0)}")
    filas = []
    for nombre, (_, crear) in MODELOS.items():
        modelo, predecir = crear(dev)
        parametros = sum(p.numel() for p in modelo.parameters())
        ms = cronometrar(predecir, imagenes)
        filas.append({"modelo": nombre, "parametros_M": round(parametros / 1e6, 2),
                      "ms_media": round(ms.mean(), 2), "ms_mediana": round(float(np.median(ms)), 2),
                      "ms_p95": round(float(np.percentile(ms, 95)), 2), "fps": round(1000 / ms.mean(), 1)})
        print(f"  {nombre:28s} {filas[-1]['fps']:6.1f} FPS  ({filas[-1]['ms_media']:.1f} ms/imagen)")
        del modelo, predecir
        torch.cuda.empty_cache()
    salida = RES / "comparacion"
    salida.mkdir(parents=True, exist_ok=True)
    with open(salida / "fps.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    actualizar_metricas()
    print(f"Guardado en {salida / 'fps.csv'}")


if __name__ == "__main__":
    main()
