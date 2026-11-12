import argparse
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

RAIZ = Path(__file__).resolve().parent
DATASET = RAIZ.parent / "dataset" / "FjordFish"
COPIA = RAIZ.parent / "dataset" / "FjordFish_T2"
SALIDA = RAIZ / "resultados" / "eda"
SEMILLA = 42
PARTICIONES = ("train", "val", "test")
CLASES = [
    "ballan", "cuckoo_female", "cuckoo_male", "corkwing_female", "corkwing_male",
    "goldsinny", "rock_cook", "labrids_unknown", "cod", "cod_yoy",
    "haddock", "pollack", "poor_cod", "saithe", "whiting",
    "gadines_unknown", "spiny_dogfish", "sharks_unknown", "brown_crab", "green_crab",
    "hermit", "lobster", "decapods_unknown", "ling", "two_spot_goby",
    "black_goby", "sand_goby", "unknown",
]
COLORES = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
TINTA, TINTA2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9, "axes.edgecolor": TINTA2,
    "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


# ---------------------------------------------------------------- 1. lectura
def leer_etiquetas(txt, problemas):
    """Lee un .txt YOLO (clase cx cy w h, normalizados). Devuelve (cajas, invalida).

    Una línea mal formada o fuera de rango invalida la imagen. Las cajas de
    ancho o alto 0 y las líneas duplicadas se descartan.
    """
    cajas, vistas = [], set()
    for n, linea in enumerate(txt.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not linea.strip():
            continue
        donde = f"{txt.parent.name}/{txt.name}:{n}"
        try:
            c, cx, cy, w, h = (float(v) for v in linea.split())
        except ValueError:
            problemas["linea_mal_formada"].append(donde)
            return [], True
        if c != int(c) or not 0 <= c < len(CLASES) or not all(0 <= v <= 1 for v in (cx, cy, w, h)):
            problemas["valor_fuera_de_rango"].append(donde)
            return [], True
        if w == 0 or h == 0:
            problemas["caja_degenerada_descartada"].append(donde)
            continue
        clave = (int(c), round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))
        if clave in vistas:
            problemas["linea_duplicada_descartada"].append(donde)
            continue
        vistas.add(clave)
        cajas.append((int(c), cx, cy, w, h))
    return cajas, False


def leer_dataset(ds, problemas):
    """Devuelve {split_original: [registro]} con tamaño, formato real y cajas de cada imagen."""
    registros = {}
    for orig in ("train", "val"):
        registros[orig] = []
        archivos = sorted((ds / "images" / orig).iterdir())
        for img in archivos:
            rel = f"images/{orig}/{img.name}"
            if img.suffix.lower() not in (".jpg", ".jpeg"):
                problemas["archivo_no_imagen_ignorado"].append(rel)
                continue
            txt = ds / "labels" / orig / f"{img.stem}.txt"
            if not txt.exists():
                problemas["imagen_sin_etiqueta_excluida"].append(rel)
                continue
            cajas, invalida = leer_etiquetas(txt, problemas)
            if invalida:
                problemas["imagen_excluida_por_etiqueta_invalida"].append(rel)
                continue
            with Image.open(img) as im:
                w, h, fmt = im.width, im.height, im.format.lower()
                if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):  # orientación EXIF rotada 90°
                    w, h = h, w
            if fmt != "jpeg":
                problemas[f"extension_jpg_pero_formato_{fmt}"].append(rel)
            registros[orig].append({"id": int(img.stem), "split_original": orig, "origen": img,
                                    "ancho_orig": w, "alto_orig": h, "formato": fmt, "cajas": cajas})
        stems = {p.stem for p in archivos}
        for txt in sorted((ds / "labels" / orig).glob("*.txt")):
            if txt.stem not in stems:
                problemas["etiqueta_sin_imagen"].append(f"labels/{orig}/{txt.name}")
    return registros


# ------------------------------------------------------------ 2. particiones
def estratificar(clases_por_item, fracciones, semilla):
    """Estratificación iterativa multietiqueta (Sechidis et al., 2011).

    Asigna cada imagen (conjunto de clases) a un subconjunto de modo que cada clase
    quede repartida según `fracciones`, partiendo por la clase más rara.
    """
    rng = random.Random(semilla)
    n, k = len(clases_por_item), len(fracciones)
    asignacion = [None] * n
    demanda_total = [f * n for f in fracciones]
    por_clase = Counter(c for cs in clases_por_item for c in cs)
    demanda = [{c: f * v for c, v in por_clase.items()} for f in fracciones]

    pendientes = {i for i, cs in enumerate(clases_por_item) if cs}
    while pendientes:
        restantes = Counter(c for i in pendientes for c in clases_por_item[i])
        minimo = min(restantes.values())
        clase = rng.choice(sorted(c for c, v in restantes.items() if v == minimo))
        idxs = sorted(i for i in pendientes if clase in clases_por_item[i])
        rng.shuffle(idxs)
        for i in idxs:
            # subconjunto que más necesita esta clase; empates por demanda total y luego azar
            mejor = max(demanda[s][clase] for s in range(k))
            cand = [s for s in range(k) if demanda[s][clase] == mejor]
            if len(cand) > 1:
                mejor_t = max(demanda_total[s] for s in cand)
                cand = [s for s in cand if demanda_total[s] == mejor_t]
            s = rng.choice(cand)
            asignacion[i] = s
            for c in clases_por_item[i]:
                demanda[s][c] -= 1
            demanda_total[s] -= 1
            pendientes.discard(i)

    vacias = [i for i in range(n) if asignacion[i] is None]
    rng.shuffle(vacias)
    for i in vacias:
        s = max(range(k), key=lambda j: (demanda_total[j], -j))
        asignacion[i] = s
        demanda_total[s] -= 1
    return asignacion


# --------------------------------------------------------- 3. copia reducida
def tamano_destino(w, h, lado_max):
    """Tamaño con lado mayor <= lado_max, conservando el aspecto y sin agrandar."""
    s = min(1.0, lado_max / max(w, h))
    return max(1, round(w * s)), max(1, round(h * s))


def convertir(tarea):
    """Guarda una imagen como JPEG reducido. Se salta si ya existe con el tamaño correcto."""
    origen, destino, (nw, nh), rehacer = tarea
    if not rehacer and destino.exists():
        try:
            with Image.open(destino) as d:
                if d.size == (nw, nh) and d.format == "JPEG":
                    return False
        except OSError:
            pass
    with Image.open(origen) as im:
        if im.format == "JPEG":
            im.draft("RGB", (nw, nh))
        im = ImageOps.exif_transpose(im).convert("RGB")
        if im.size != (nw, nh):
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        tmp = destino.with_suffix(".tmp")
        im.save(tmp, "JPEG", quality=95, subsampling=0, optimize=True)
    os.replace(tmp, destino)
    return True


def a_coco(regs, split, copia):
    """Anotaciones COCO en píxeles de la copia reducida. category_id = clase + 1."""
    imagenes, anotaciones = [], []
    for r in regs:
        W, H = r["ancho"], r["alto"]
        imagenes.append({"id": r["id"], "file_name": f"images/{split}/{r['id']}.jpg",
                         "width": W, "height": H})
        for c, cx, cy, w, h in r["cajas"]:
            x1, y1 = max(0.0, (cx - w / 2) * W), max(0.0, (cy - h / 2) * H)
            x2, y2 = min(W, (cx + w / 2) * W), min(H, (cy + h / 2) * H)
            anotaciones.append({"id": len(anotaciones) + 1, "image_id": r["id"], "category_id": c + 1,
                                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                                "area": round((x2 - x1) * (y2 - y1), 2), "iscrowd": 0})
    return {"info": {"description": f"FjordFish - {split} (Tarea 2 DL)", "dataset_dir": copia.as_posix()},
            "images": imagenes, "annotations": anotaciones,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASES)]}


def crear_copia(parts, copia, lado_max, rehacer):
    t0 = time.time()
    tareas, esperados = [], set()
    for s, regs in parts.items():
        for sub in ("images", "labels"):
            (copia / sub / s).mkdir(parents=True, exist_ok=True)
        for r in regs:
            r["ancho"], r["alto"] = tamano_destino(r["ancho_orig"], r["alto_orig"], lado_max)
            img, txt = copia / "images" / s / f"{r['id']}.jpg", copia / "labels" / s / f"{r['id']}.txt"
            tareas.append((r["origen"], img, (r["ancho"], r["alto"]), rehacer))
            txt.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                                   for c, cx, cy, w, h in r["cajas"]), encoding="utf-8")
            esperados |= {img, txt}
    sobrantes = [p for s in PARTICIONES for sub in ("images", "labels")
                 for p in (copia / sub / s).iterdir() if p.is_file() and p not in esperados]
    for p in sobrantes:
        p.unlink()

    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        nuevas = sum(ex.map(convertir, tareas, chunksize=16))
    print(f"Copia reducida: {len(tareas)} imágenes ({nuevas} nuevas, {len(sobrantes)} sobrantes "
          f"eliminadas) en {time.time() - t0:.0f} s -> {copia}")

    (copia / "coco").mkdir(exist_ok=True)
    for s, regs in parts.items():
        (copia / "coco" / f"{s}.json").write_text(json.dumps(a_coco(regs, s, copia)), encoding="utf-8")
    # sin `path:`, Ultralytics usa como raíz la carpeta del yaml
    (copia / "fjordfish.yaml").write_text(
        "# Generado por Tarea 2/eda.py. Rutas relativas a la carpeta de este archivo.\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASES)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASES)),
        encoding="utf-8")


def verificar(parts, copia):
    """Comprueba que imágenes, etiquetas YOLO y COCO de la copia coincidan con las particiones."""
    ok = True
    for s, regs in parts.items():
        coco = json.loads((copia / "coco" / f"{s}.json").read_text(encoding="utf-8"))
        n_img = len(list((copia / "images" / s).glob("*.jpg")))
        n_yolo = sum(len(t.read_text(encoding="utf-8").splitlines())
                     for t in (copia / "labels" / s).glob("*.txt"))
        n_cajas = sum(len(r["cajas"]) for r in regs)
        bien = n_img == len(regs) == len(coco["images"]) and n_yolo == len(coco["annotations"]) == n_cajas
        ok &= bien
        print(f"  {s:5s}: {n_img} imágenes, {n_yolo} cajas YOLO, {len(coco['annotations'])} cajas COCO "
              f"(esperado {len(regs)} / {n_cajas}) -> {'OK' if bien else 'FALLA'}")
    try:
        from ultralytics.data.utils import check_det_dataset
        d = check_det_dataset(str(copia / "fjordfish.yaml"))
        print(f"  Ultralytics lee fjordfish.yaml: OK (nc={d['nc']})")
    except ImportError:
        print("  Ultralytics no instalado: se omite su chequeo")
    print("Verificación:", "OK" if ok else "CON FALLAS")
    return ok


# ------------------------------------------------------------------ 4. EDA
def tablas_eda(parts):
    """DataFrames de imágenes y de cajas, en resolución original."""
    imgs, cajas = [], []
    for s, regs in parts.items():
        for r in regs:
            W, H = r["ancho_orig"], r["alto_orig"]
            imgs.append({"id": r["id"], "split": s, "ancho": W, "alto": H})
            for c, cx, cy, w, h in r["cajas"]:
                bw = min(1, cx + w / 2) - max(0, cx - w / 2)
                bh = min(1, cy + h / 2) - max(0, cy - h / 2)
                cajas.append({"id": r["id"], "split": s, "clase_id": c, "clase": CLASES[c],
                              "area_rel": bw * bh, "aspecto": (bw * W) / (bh * H)})
    return pd.DataFrame(imgs), pd.DataFrame(cajas)


def tabla_clases(imgs, cajas):
    tab = pd.DataFrame({"clase_id": range(len(CLASES)), "clase": CLASES})
    for s in (*PARTICIONES, "total"):
        a = cajas if s == "total" else cajas[cajas["split"] == s]
        tab[f"{s}_imagenes"] = tab["clase_id"].map(a.groupby("clase_id")["id"].nunique()).fillna(0).astype(int)
        tab[f"{s}_instancias"] = tab["clase_id"].map(a.groupby("clase_id").size()).fillna(0).astype(int)
    total = {"clase_id": "", "clase": "TOTAL"}
    for s in (*PARTICIONES, "total"):
        a_img = imgs if s == "total" else imgs[imgs["split"] == s]
        total[f"{s}_imagenes"] = len(a_img)
        total[f"{s}_instancias"] = int(tab[f"{s}_instancias"].sum())
    return pd.concat([tab, pd.DataFrame([total])], ignore_index=True)


def guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def grafico_clases(tab, ruta):
    orden = tab[tab["clase"] != "TOTAL"].sort_values("total_instancias")
    y, h = np.arange(len(orden)), 0.27
    fig, ax = plt.subplots(figsize=(8, 9))
    for i, s in enumerate(PARTICIONES):
        v = orden[f"{s}_instancias"].to_numpy(dtype=float)
        ax.barh(y + (1 - i) * h, np.where(v > 0, v, np.nan), height=h - 0.03, color=COLORES[s], label=s)
    ax.set_xscale("log")
    ax.set_yticks(y, [f"{n} ({c})" for n, c in zip(orden["clase"], orden["clase_id"])])
    ax.set(xlabel="Instancias (escala log)", title="Instancias por clase y partición")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    guardar(fig, ruta)


def grafico_objetos(imgs, cajas, ruta):
    n_obj = imgs["id"].map(cajas.groupby("id").size()).fillna(0).astype(int)
    bins = np.arange(0, n_obj.max() + 2) - 0.5
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True)
    for ax, s in zip(axes, PARTICIONES):
        d = n_obj[imgs["split"] == s]
        ax.hist(d, bins=bins, color=COLORES[s], rwidth=0.85)
        ax.set(title=f"{s} (n={len(d)}, media={d.mean():.2f}, mediana={d.median():.0f})",
               xlabel="Objetos por imagen")
    axes[0].set_ylabel("Imágenes")
    guardar(fig, ruta)


def grafico_cajas(cajas, ruta):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    b_area = np.logspace(np.log10(max(cajas["area_rel"].min(), 1e-6)), 0, 50)
    b_asp = np.logspace(np.log10(cajas["aspecto"].min()), np.log10(cajas["aspecto"].max()), 50)
    for s in PARTICIONES:
        a = cajas[cajas["split"] == s]
        kw = dict(density=True, histtype="step", linewidth=2, color=COLORES[s], label=s)
        axes[0].hist(a["area_rel"], bins=b_area, **kw)
        axes[1].hist(a["aspecto"], bins=b_asp, **kw)
    ref = 32 ** 2 / 640 ** 2
    axes[0].axvline(ref, color=TINTA2, linewidth=1, linestyle="--")
    axes[0].text(ref, axes[0].get_ylim()[1] * 0.95, " 32 px a 640", color=TINTA2, fontsize=8, va="top")
    axes[0].set(xscale="log", xlabel="Área relativa de la caja (w*h / W*H, escala log)",
                ylabel="Densidad", title="Tamaño de las cajas")
    axes[1].axvline(1, color=TINTA2, linewidth=1, linestyle="--")
    axes[1].set(xscale="log", xlabel="Relación de aspecto w/h (escala log)",
                title="Relación de aspecto de las cajas")
    axes[0].legend()
    guardar(fig, ruta)


def grafico_resoluciones(imgs, ruta_png, ruta_csv):
    res = imgs["ancho"].astype(str) + "x" + imgs["alto"].astype(str)
    tab = imgs.assign(resolucion=res).groupby(["resolucion", "split"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=list(PARTICIONES), fill_value=0)
    tab["total"] = tab.sum(axis=1)
    tab = tab.sort_values("total", ascending=False)
    tab.to_csv(ruta_csv)
    top, y, h = tab.head(10).iloc[::-1], np.arange(min(10, len(tab))), 0.27
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    for i, s in enumerate(PARTICIONES):
        axes[0].barh(y + (1 - i) * h, top[s], height=h - 0.03, color=COLORES[s], label=s)
        d = imgs[imgs["split"] == s]
        axes[1].scatter(d["ancho"], d["alto"], s=12, alpha=0.5, color=COLORES[s], edgecolors="none", label=s)
    axes[0].set_yticks(y, top.index)
    axes[0].set(xlabel="Imágenes", title=f"10 resoluciones más frecuentes (otras {max(0, len(tab) - 10)}: "
                                         f"{int(tab.iloc[10:]['total'].sum())} imágenes)")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", visible=False)
    axes[1].set(xlabel="Ancho (px)", ylabel="Alto (px)", title="Ancho vs alto")
    axes[1].legend(loc="upper left")
    guardar(fig, ruta_png)
    return tab


def grafico_ejemplos(cajas, parts, copia, ruta, n=9, semilla=SEMILLA):
    """Cuadrícula de imágenes de train con sus cajas, priorizando la variedad de clases."""
    rng = random.Random(semilla)
    por_img = cajas[cajas["split"] == "train"].groupby("id")["clase_id"].apply(set).to_dict()
    ids = list(por_img)
    rng.shuffle(ids)
    elegidas, cubiertas = [], set()
    for _ in range(min(n, len(ids))):
        i = max((i for i in ids if i not in elegidas), key=lambda i: len(por_img[i] - cubiertas))
        elegidas.append(i)
        cubiertas |= por_img[i]

    regs = {r["id"]: r for r in parts["train"]}
    paleta = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fuente = ImageFont.load_default()
    filas = int(np.ceil(n / 3))
    fig, axes = plt.subplots(filas, 3, figsize=(3 * 4.6, filas * 3.0))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), elegidas):
        r = regs[i]
        im = Image.open(copia / "images" / "train" / f"{i}.jpg").convert("RGB")
        im.thumbnail((900, 900))
        W, H, dib = im.width, im.height, ImageDraw.Draw(im)
        for c, cx, cy, w, h in r["cajas"]:
            x0, y0 = max(0, (cx - w / 2) * W), max(0, (cy - h / 2) * H)
            col = paleta[c % len(paleta)]
            dib.rectangle([x0, y0, min(W, (cx + w / 2) * W), min(H, (cy + h / 2) * H)], outline=col, width=3)
            tb = dib.textbbox((x0, y0), CLASES[c], font=fuente)
            ty = max(0, y0 - (tb[3] - tb[1]) - 4)
            dib.rectangle([x0, ty, x0 + tb[2] - tb[0] + 4, ty + tb[3] - tb[1] + 4], fill=col)
            dib.text((x0 + 2, ty + 1), CLASES[c], fill="white", font=fuente)
        ax.imshow(im)
        ax.set_title(f"images/{r['split_original']}/{i}.jpg  ({r['ancho_orig']}x{r['alto_orig']})", fontsize=8)
    guardar(fig, ruta)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path, default=DATASET, help="FjordFish original (images/, labels/)")
    ap.add_argument("--copia", type=Path, default=COPIA, help="Destino de la copia reducida")
    ap.add_argument("--salida", type=Path, default=SALIDA, help="Carpeta de resultados del EDA")
    ap.add_argument("--val", type=float, default=0.15, help="Fracción del train original que va a val")
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    ap.add_argument("--lado-max", type=int, default=1280, help="Lado mayor de la copia reducida (px)")
    ap.add_argument("--rehacer", action="store_true", help="Regenerar las imágenes ya convertidas")
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    problemas = defaultdict(list)
    registros = leer_dataset(args.dataset.resolve(), problemas)

    train_orig = sorted(registros["train"], key=lambda r: r["id"])
    asignacion = estratificar([{b[0] for b in r["cajas"]} for r in train_orig],
                              [1 - args.val, args.val], args.semilla)
    parts = {"train": [r for r, a in zip(train_orig, asignacion) if a == 0],
             "val": [r for r, a in zip(train_orig, asignacion) if a == 1],
             "test": sorted(registros["val"], key=lambda r: r["id"])}

    crear_copia(parts, args.copia.resolve(), args.lado_max, args.rehacer)
    ok = verificar(parts, args.copia.resolve())

    out = args.salida
    with open(out / "particiones.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "particion", "split_original", "formato_original", "ancho_orig", "alto_orig",
                     "ancho", "alto", "escala_x", "escala_y", "n_objetos", "clases"])
        for s, regs in parts.items():
            for r in regs:
                wr.writerow([r["id"], s, r["split_original"], r["formato"], r["ancho_orig"], r["alto_orig"],
                             r["ancho"], r["alto"], f"{r['ancho'] / r['ancho_orig']:.6f}",
                             f"{r['alto'] / r['alto_orig']:.6f}", len(r["cajas"]),
                             " ".join(map(str, sorted({b[0] for b in r["cajas"]})))])
    (out / "problemas_dataset.json").write_text(json.dumps({
        "conteos": {k: len(v) for k, v in sorted(problemas.items())},
        "ejemplos": {k: v[:10] for k, v in sorted(problemas.items())},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    imgs, cajas = tablas_eda(parts)
    tab = tabla_clases(imgs, cajas)
    tab.to_csv(out / "clases_por_particion.csv", index=False)
    (cajas.groupby(["clase_id", "clase"])
     .agg(instancias=("area_rel", "size"), area_rel_mediana=("area_rel", "median"),
          area_rel_p10=("area_rel", lambda v: v.quantile(0.1)),
          area_rel_p90=("area_rel", lambda v: v.quantile(0.9)), aspecto_mediana=("aspecto", "median"))
     .reset_index().to_csv(out / "estadisticas_cajas.csv", index=False, float_format="%.5f"))
    grafico_clases(tab, out / "distribucion_clases.png")
    grafico_objetos(imgs, cajas, out / "objetos_por_imagen.png")
    grafico_cajas(cajas, out / "tamano_cajas.png")
    grafico_resoluciones(imgs, out / "resoluciones.png", out / "resoluciones.csv")
    grafico_ejemplos(cajas, parts, args.copia.resolve(), out / "ejemplos.png")

    print("Problemas del dataset:", ", ".join(f"{k}={len(v)}" for k, v in sorted(problemas.items())))
    for s in PARTICIONES:
        print(f"  {s:5s}: {len(parts[s]):5d} imágenes, {sum(len(r['cajas']) for r in parts[s]):5d} objetos")
    vacias = tab[(tab["clase"] != "TOTAL") & (tab["total_instancias"] == 0)]["clase"].tolist()
    print("Clases sin objetos:", ", ".join(vacias) or "ninguna")
    print(f"Resultados del EDA en {out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
