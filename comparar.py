import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

RAIZ = Path(__file__).resolve().parent
COPIA = RAIZ.parent / "dataset" / "FjordFish_T2"
RES = RAIZ / "resultados"
SALIDA = RES / "comparacion"

# nombre -> (carpeta en resultados/, color en los gráficos)
MODELOS = {
    "CNN propia (cero)": ("cnn_propia", "#2a78d6"),
    "YOLO11s (cero)": ("yolo_cero", "#eb6834"),
    "Faster R-CNN (COCO)": ("faster_rcnn", "#1baf7a"),
    "YOLO11s (COCO, congelado)": ("yolo_coco_congelado", "#eda100"),
    "YOLO11s (COCO, completo)": ("yolo_coco", "#e87ba4"),
}
TINTA, TINTA2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRILLA, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
    "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRILLA, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 150,
})


# ---------------------------------------------------------------- lectura
def leer_csv(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [{k.strip(): v for k, v in fila.items()} for fila in csv.DictReader(f)]


def cargar():
    fps = {f["modelo"]: f for f in leer_csv(SALIDA / "fps.csv")}
    datos = {}
    for nombre, (carpeta, _) in MODELOS.items():
        m = json.loads((RES / carpeta / "metricas_test.json").read_text(encoding="utf-8"))
        m["fps"] = float(fps[nombre]["fps"])
        m["ms_por_imagen"] = float(fps[nombre]["ms_media"])
        m["ap_clase"] = {f["clase"]: f for f in leer_csv(RES / carpeta / "ap_por_clase_test.csv")}
        m["historial"] = historial(carpeta)
        datos[nombre] = m
    return datos


def historial(carpeta):
    """(épocas, val mAP50-95, minutos acumulados) de cada entrenamiento."""
    if carpeta.startswith("yolo"):
        filas = leer_csv(RES / carpeta / "entrenamiento" / "results.csv")
        ep = [int(f["epoch"]) for f in filas]
        mapa = [float(f["metrics/mAP50-95(B)"]) for f in filas]
        minutos = [float(f["time"]) / 60 for f in filas]
    else:
        filas = leer_csv(RES / carpeta / "historial.csv")
        ep = [int(f["epoca"]) for f in filas]
        mapa = [float(f["val_mAP50_95"]) for f in filas]
        minutos = list(np.cumsum([float(f["seg_epoca"]) for f in filas]) / 60)
    return np.array(ep), np.array(mapa), np.array(minutos)


def escalar_minutos(datos):
    """Reescala los minutos acumulados para que el total coincida con el tiempo de la tabla."""
    for m in datos.values():
        ep, mapa, minutos = m["historial"]
        m["historial"] = (ep, mapa, minutos * m["tiempo_entrenamiento_min"] / minutos[-1])


# ---------------------------------------------------------------- tabla
# (clave, encabezado, decimales)
COLUMNAS = [
    ("mAP50_95", "mAP50-95", 3), ("mAP50", "mAP50", 3), ("mAP75", "mAP75", 3),
    ("mAP_pequenos", "mAP pequeños", 3), ("mAP_medianos", "mAP medianos", 3), ("mAP_grandes", "mAP grandes", 3),
    ("precision", "Precisión", 3), ("recall", "Recall", 3), ("F1", "F1", 3), ("umbral_conf", "Umbral conf.", 2),
    ("parametros_M", "Parámetros (M)", 2), ("entrenables_M", "Entrenables (M)", 2),
    ("tiempo_entrenamiento_min", "Entrenamiento (min)", 1), ("epocas", "Épocas", 0),
    ("mejor_epoca", "Mejor época", 0), ("fps", "FPS", 1), ("ms_por_imagen", "ms/imagen", 1),
]


def tabla(datos):
    filas = []
    for nombre, m in datos.items():
        m["parametros_M"] = m["parametros"] / 1e6
        m["entrenables_M"] = m.get("parametros_entrenables", m["parametros"]) / 1e6
        filas.append({"Modelo": nombre, **{enc: round(m[k], d) if d else int(m[k]) for k, enc, d in COLUMNAS}})
    with open(SALIDA / "tabla_final.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    return filas


# ---------------------------------------------------------------- gráficos
def barras_agrupadas(ax, datos, claves, etiquetas, titulo):
    n = len(datos)
    ancho = 0.8 / n
    x = np.arange(len(claves))
    for i, (nombre, m) in enumerate(datos.items()):
        vals = [m[k] for k in claves]
        barras = ax.bar(x + (i - (n - 1) / 2) * ancho, vals, ancho * 0.92, color=MODELOS[nombre][1],
                        label=nombre, edgecolor="white", linewidth=0.5)
        for b, v in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color=TINTA2, rotation=90)
    ax.set_xticks(x, etiquetas)
    ax.set_title(titulo, loc="left")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(m[k] for m in datos.values() for k in claves) * 1.25)


def grafico_metricas(datos):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1, 1]})
    barras_agrupadas(a1, datos, ["mAP50_95", "mAP50", "mAP75"], ["mAP50-95", "mAP50", "mAP75"],
                     "mAP en test (más alto = mejor)")
    barras_agrupadas(a2, datos, ["precision", "recall", "F1"], ["Precisión", "Recall", "F1"],
                     "Precisión, recall y F1 (IoU 0.5, umbral de máximo F1)")
    a1.legend(loc="upper center", bbox_to_anchor=(1.05, -0.1), ncol=5, fontsize=9)
    fig.savefig(SALIDA / "metricas_globales.png")
    plt.close(fig)


def grafico_tamanos(datos):
    fig, ax = plt.subplots(figsize=(8, 4.3))
    barras_agrupadas(ax, datos, ["mAP_pequenos", "mAP_medianos", "mAP_grandes"],
                     ["Pequeños (< 32² px)", "Medianos (32²-96² px)", "Grandes (> 96² px)"],
                     "mAP50-95 en test según el tamaño del objeto")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
    fig.savefig(SALIDA / "map_por_tamano.png")
    plt.close(fig)


# desplazamiento de las etiquetas que se solapan
DESPLAZAMIENTO = {
    "fps": {"Faster R-CNN (COCO)": (-10, 14), "YOLO11s (COCO, congelado)": (10, 2), "YOLO11s (cero)": (10, -12)},
    "tiempo_entrenamiento_min": {"Faster R-CNN (COCO)": (-40, 16), "YOLO11s (cero)": (-30, -20)},
}


def grafico_eficiencia(datos):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, clave, xlabel, titulo in (
            (a1, "fps", "FPS (lote 1, más alto = más rápido)", "Exactitud contra velocidad de inferencia"),
            (a2, "tiempo_entrenamiento_min", "Tiempo de entrenamiento (min)", "Exactitud contra costo de entrenamiento")):
        for nombre, m in datos.items():
            tam = 40 + m["parametros"] / 1e6 * 12
            ax.scatter(m[clave], m["mAP50_95"], s=tam, color=MODELOS[nombre][1], edgecolor="white",
                       linewidth=2, zorder=3)
            ax.annotate(nombre, (m[clave], m["mAP50_95"]), textcoords="offset points",
                        xytext=DESPLAZAMIENTO[clave].get(nombre, (8, 6)), fontsize=8.5, color=TINTA)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mAP50-95 en test")
        ax.set_title(titulo, loc="left")
        ax.set_ylim(0, max(m["mAP50_95"] for m in datos.values()) * 1.25)
        ax.set_xlim(0, max(m[clave] for m in datos.values()) * 1.35)
    fig.text(0.5, -0.02, "El tamaño del punto es proporcional a la cantidad de parámetros "
             "(Faster R-CNN 41,4 M, CNN propia 12,0 M, YOLO11s 9,4 M).", ha="center", color=TINTA2, fontsize=9)
    fig.savefig(SALIDA / "eficiencia.png")
    plt.close(fig)


def grafico_curvas(datos):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for nombre, m in datos.items():
        ep, mapa, minutos = m["historial"]
        color = MODELOS[nombre][1]
        a1.plot(ep, mapa, color=color, linewidth=2, label=nombre)
        a2.plot(minutos, mapa, color=color, linewidth=2, label=nombre)
        mejor = int(np.argmax(mapa))
        for ax, x in ((a1, ep[mejor]), (a2, minutos[mejor])):
            ax.scatter(x, mapa[mejor], s=45, color=color, edgecolor="white", linewidth=2, zorder=3)
    a1.set_xlabel("Época")
    a2.set_xlabel("Minutos de entrenamiento")
    a1.set_ylabel("mAP50-95 en val")
    a1.set_title("Convergencia por época (punto = mejor época)", loc="left")
    a2.set_title("Convergencia por tiempo de entrenamiento", loc="left")
    a1.set_ylim(0, None)
    a1.legend(loc="upper center", bbox_to_anchor=(1.03, -0.15), ncol=5, fontsize=8.5)
    fig.savefig(SALIDA / "curvas_val.png")
    plt.close(fig)


def grafico_ap_clase(datos):
    primero = next(iter(datos.values()))["ap_clase"]
    clases = sorted((c for c, f in primero.items() if int(f["objetos"]) > 0),
                    key=lambda c: -int(primero[c]["objetos"]))
    mat = np.array([[float(m["ap_clase"][c]["AP50"]) for m in datos.values()] for c in clases])
    with open(SALIDA / "ap_por_clase.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase", "objetos_test", *datos])
        for c, fila in zip(clases, mat):
            w.writerow([c, primero[c]["objetos"], *[round(v, 3) for v in fila]])

    azul = matplotlib.colors.LinearSegmentedColormap.from_list(
        "azul", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    fig, ax = plt.subplots(figsize=(8.5, 10))
    ax.imshow(mat, cmap=azul, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.55 else TINTA)
    ax.set_xticks(range(len(datos)), [n.replace(" (", "\n(") for n in datos], fontsize=8.5)
    ax.set_yticks(range(len(clases)), [f"{c} ({primero[c]['objetos']})" for c in clases], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("AP50 por clase en test (entre paréntesis: objetos en test; ordenadas de más a menos)",
                 loc="left", pad=40, fontsize=10)
    fig.savefig(SALIDA / "ap_por_clase.png")
    plt.close(fig)


# ---------------------------------------------------------------- ejemplos
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def elegir_imagenes(coco, n=4, semilla=42):
    """Imágenes de test con 2-6 objetos no diminutos y de clases variadas."""
    por_img = {}
    for a in coco["annotations"]:
        por_img.setdefault(a["image_id"], []).append(a)
    info = {im["id"]: im for im in coco["images"]}
    candidatas = [i for i, anns in por_img.items() if 2 <= len(anns) <= 6 and
                  np.median([a["bbox"][2] * a["bbox"][3] / (info[i]["width"] * info[i]["height"])
                             for a in anns]) > 0.004]
    rng = np.random.default_rng(semilla)
    rng.shuffle(candidatas)
    elegidas, clases_vistas = [], set()
    for i in candidatas:
        clases = {a["category_id"] for a in por_img[i]}
        if not clases <= clases_vistas:
            elegidas.append(i)
            clases_vistas |= clases
        if len(elegidas) == n:
            break
    return [info[i] for i in elegidas], por_img


def grafico_ejemplos(datos):
    coco = json.loads((COPIA / "coco" / "test.json").read_text(encoding="utf-8"))
    nombres = {c["id"]: c["name"] for c in coco["categories"]}
    imagenes, por_img = elegir_imagenes(coco)
    ids = {im["id"] for im in imagenes}
    preds = {}
    for nombre, (carpeta, _) in MODELOS.items():
        todas = json.loads((RES / carpeta / "predicciones_test.json").read_text(encoding="utf-8"))
        umbral = datos[nombre]["umbral_conf"]
        preds[nombre] = [p for p in todas if p["image_id"] in ids and p["score"] >= umbral]

    BIEN, MAL, GT = "#22c55e", "#ef4444", "#ffffff"
    cols = ["Etiquetas reales", *MODELOS]
    fig, axes = plt.subplots(len(imagenes), len(cols), figsize=(3.2 * len(cols), 1.95 * len(imagenes)))
    for r, im in enumerate(imagenes):
        img = cv2.imread(str(COPIA / im["file_name"]))[:, :, ::-1]
        gts = [(xyxy(a["bbox"]), a["category_id"]) for a in por_img[im["id"]]]
        for c, col in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(col.replace(" (", "\n("), fontsize=9)
            if c == 0:
                cajas = [(b, k, GT, None) for b, k in gts]
                ax.set_ylabel(f"id {im['id']}", fontsize=8, color=TINTA2)
            else:
                cajas, hallados = [], set()
                for p in sorted((q for q in preds[col] if q["image_id"] == im["id"]), key=lambda q: -q["score"]):
                    b = xyxy(p["bbox"])
                    ok = [j for j, (g, k) in enumerate(gts)
                          if j not in hallados and k == p["category_id"] and iou(b, g) >= 0.5]
                    if ok:
                        hallados.add(ok[0])
                    cajas.append((b, p["category_id"], BIEN if ok else MAL, p["score"]))
                # objetos reales no detectados
                for j, (g, _) in enumerate(gts):
                    if j not in hallados:
                        ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False,
                                               edgecolor=GT, linewidth=1.2, linestyle=(0, (2, 2))))
            for (x1, y1, x2, y2), k, color, s in cajas:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.6))
                texto = nombres[k] if s is None else f"{nombres[k]} {s:.2f}"
                ax.text(x1, y1 - 2, texto, fontsize=5.5, color="black", va="bottom",
                        bbox=dict(facecolor=color, edgecolor="none", pad=0.6, alpha=0.85))
    fig.legend(handles=[Patch(facecolor="white", edgecolor="black", label="Etiqueta real"),
                        Patch(color=BIEN, label="Correcta (misma clase, IoU ≥ 0.5)"),
                        Patch(color=MAL, label="Incorrecta (clase o posición equivocada)"),
                        Patch(facecolor="none", edgecolor="#888888", linestyle="--",
                              label="Objeto real no detectado")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03), fontsize=9)
    fig.suptitle("Predicciones en test (se muestran las cajas sobre el umbral de máximo F1 de cada modelo)",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(SALIDA / "ejemplos_predicciones.png")
    plt.close(fig)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    datos = cargar()
    filas = tabla(datos)
    escalar_minutos(datos)
    grafico_metricas(datos)
    grafico_tamanos(datos)
    grafico_eficiencia(datos)
    grafico_curvas(datos)
    grafico_ap_clase(datos)
    grafico_ejemplos(datos)
    print(f"{'Modelo':28s} {'mAP50-95':>8s} {'mAP50':>6s} {'P':>5s} {'R':>5s} {'FPS':>6s} {'min':>5s}")
    for f in filas:
        print(f"{f['Modelo']:28s} {f['mAP50-95']:8.3f} {f['mAP50']:6.3f} {f['Precisión']:5.2f} "
              f"{f['Recall']:5.2f} {f['FPS']:6.1f} {f['Entrenamiento (min)']:5.1f}")
    print(f"Guardado en {SALIDA}")


if __name__ == "__main__":
    main()
