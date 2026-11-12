import argparse
import contextlib
import csv
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

RAIZ = Path(__file__).resolve().parent
COCO_DIR = RAIZ.parent / "dataset" / "FjordFish_T2" / "coco"
RESULTADOS = RAIZ / "resultados"
METRICAS = ["mAP50_95", "mAP50", "mAP75", "mAP_pequenos", "mAP_medianos", "mAP_grandes"]


def cargar_gt(split):
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(COCO_DIR / f"{split}.json"))


def _precision_recall(ev, ids_clase):
    """Precisión, recall y F1 a IoU 0.5 con el emparejamiento de pycocotools.

    Devuelve el resultado global y el de cada clase en el umbral de máximo F1.
    """
    todas = ev.params.areaRng[0]
    max_det = ev.params.maxDets[-1]
    scores, aciertos, clases, n_gt = [], [], [], Counter()
    for e in ev.evalImgs:
        if e is None or e["aRng"] != todas or e["maxDet"] != max_det:
            continue
        n_gt[e["category_id"]] += int((~np.asarray(e["gtIgnore"], bool)).sum())
        if len(e["dtScores"]):
            validas = ~np.asarray(e["dtIgnore"])[0].astype(bool)
            scores += list(np.asarray(e["dtScores"])[validas])
            aciertos += list((np.asarray(e["dtMatches"])[0] > 0)[validas])
            clases += [e["category_id"]] * int(validas.sum())
    total_gt = sum(n_gt.values())
    if not scores or not total_gt:
        return {"precision": 0.0, "recall": 0.0, "F1": 0.0, "umbral_conf": None}, {}
    orden = np.argsort(-np.asarray(scores), kind="mergesort")
    s, tp, c = np.asarray(scores)[orden], np.asarray(aciertos)[orden], np.asarray(clases)[orden]
    tp_acum = np.cumsum(tp)
    prec = tp_acum / np.arange(1, len(tp) + 1)
    rec = tp_acum / total_gt
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    k = int(np.argmax(f1))
    glob = {"precision": float(prec[k]), "recall": float(rec[k]), "F1": float(f1[k]),
            "umbral_conf": float(s[k])}
    por_clase = {}
    for cid in ids_clase:
        sel = (c == cid) & (s >= s[k])
        n_tp = int(tp[sel].sum())
        por_clase[cid] = {"precision": n_tp / sel.sum() if sel.sum() else 0.0,
                          "recall": n_tp / n_gt[cid] if n_gt[cid] else float("nan")}
    return glob, por_clase


def evaluar(predicciones, split="test", gt=None):
    gt = gt or cargar_gt(split)
    ids_img = gt.getImgIds()
    ids_clase = gt.getCatIds()
    nombres = {c["id"]: c["name"] for c in gt.loadCats(ids_clase)}
    n_obj = Counter(a["category_id"] for a in gt.loadAnns(gt.getAnnIds(imgIds=ids_img)))
    validos = set(ids_img)
    preds = [dict(p) for p in predicciones if p["image_id"] in validos]
    res = {"particion": split, "imagenes": len(ids_img), "objetos": sum(n_obj.values()),
           "predicciones": len(preds), "clases_evaluadas": sum(1 for v in n_obj.values() if v)}
    if not preds:
        res.update({k: 0.0 for k in METRICAS})
        res.update({"precision": 0.0, "recall": 0.0, "F1": 0.0, "umbral_conf": None, "por_clase": []})
        return res

    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(preds), "bbox")
        ev.params.imgIds = ids_img
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    res.update({k: max(0.0, float(v)) for k, v in zip(METRICAS, ev.stats[:6])})
    glob, pr_clase = _precision_recall(ev, ids_clase)
    res.update(glob)

    # ev.eval["precision"]: [IoU, recall, clase, área, maxDets], con -1 donde no hay objetos
    prec = ev.eval["precision"]
    res["por_clase"] = []
    for k, cid in enumerate(ev.params.catIds):
        p_all, p50 = prec[:, :, k, 0, -1], prec[0, :, k, 0, -1]
        res["por_clase"].append({
            "clase": nombres[cid], "objetos": n_obj.get(cid, 0),
            "AP50_95": float(p_all[p_all > -1].mean()) if (p_all > -1).any() else None,
            "AP50": float(p50[p50 > -1].mean()) if (p50 > -1).any() else None,
            "precision": pr_clase.get(cid, {}).get("precision"),
            "recall": pr_clase.get(cid, {}).get("recall") if n_obj.get(cid) else None,
        })
    return res


def guardar(res, carpeta, extra=None):
    """Escribe metricas_<split>.json y ap_por_clase_<split>.csv; extra agrega datos del modelo."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    s = res["particion"]
    salida = {**(extra or {}), **{k: v for k, v in res.items() if k != "por_clase"}}
    (carpeta / f"metricas_{s}.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    with open(carpeta / f"ap_por_clase_{s}.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["clase", "objetos", "AP50_95", "AP50", "precision", "recall"])
        wr.writeheader()
        for fila in res["por_clase"]:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fila.items()})


def resumen(res, nombre=""):
    return (f"{nombre:<28s} mAP50-95={res['mAP50_95']:.3f}  mAP50={res['mAP50']:.3f}  "
            f"P={res['precision']:.3f}  R={res['recall']:.3f}  F1={res['F1']:.3f}  "
            f"({res['predicciones']} predicciones)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predicciones", type=Path, help="JSON con predicciones en formato COCO")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--nombre", help="carpeta del modelo en resultados/ donde guardar las métricas")
    args = ap.parse_args()
    res = evaluar(json.loads(args.predicciones.read_text(encoding="utf-8")), args.split)
    print(resumen(res, args.nombre or args.predicciones.stem))
    if args.nombre:
        guardar(res, RESULTADOS / args.nombre)


if __name__ == "__main__":
    main()
