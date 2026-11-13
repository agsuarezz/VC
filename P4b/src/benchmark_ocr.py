import os
import re
import glob
import time
import cv2
import torch
import easyocr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from difflib import SequenceMatcher
from transformers import AutoProcessor, AutoModelForImageTextToText

# Configuración estética de gráficas
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
P4_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "P4"))
DATASET_DIR = os.path.join(P4_DIR, "dataset")
IMAGES_OUT_P4B = os.path.join(BASE_DIR, "images")
IMAGES_OUT_P4 = os.path.join(P4_DIR, "images")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGES_OUT_P4B, exist_ok=True)
os.makedirs(IMAGES_OUT_P4, exist_ok=True)


# Anotaciones manuales completadas y verificadas visualmente sobre el split de test (donde faltaba el texto)
MANUAL_TEST_ANNOTATIONS = {
    "mgz-42.jpg": "GC1808CK",
    "mgz-29.jpg": "1897JYL",
    "alvaro_34.jpg": "6727HYC",
    "mgz-32.jpg": "7016CRG",
    "mgz-05.jpg": "7498KSF",
    "mgz-39.jpg": "7486GLL",
    "mgz-20.jpg": "1551MTB",
    "mgz-40.jpg": "9240KPN",
    "mgz-22.jpg": "7796KWR",
    "mgz-41.jpg": "9762NGT",
    "mgz-30.jpg": "3659MNV",
    "mgz-14.jpg": "1128FNG",
    "alvaro_44.jpg": "5679LGW",
    "alvaro_48.jpg": "5376KGD",
    "IMG_5934.jpeg": "8409DRN",
    "si38.jpg": "GC1002CB",
    "IMG_5966.jpeg": "1095MYD",
    "matricula_eso_43.jpg": "0549HZV",
    "matricula_eso_11.jpg": "7203BWN",
    "61F803C9-BF37-43B8-8AFC-4AE891E39B17.JPG": "1154NBW",
    "IMG20261102123149.jpg": "3292GRL",
    "cocheE21.jpg": "0707LTF",
    "GC6657BT.JPG": "GC6657BT",
    "IMG_0015.jpg": "1966HYX",
    "coche_sbs_009.jpeg": "1306FPR",
    "matriculita13.jpg": "0170KJN",
    "IMG_6570.jpg": "3867LWH",
    "20261031_1540_2.jpg": "8307MGF",
    "IMG_6550.jpg": "3215JDY",
    "20261031_1540_68.jpg": "5168LMM",
    "coche_sbs_007.jpeg": "4773CTK",
    "IMG-20261031-WA0017.jpg": "9825CBC",
    "matricula_eso_34.jpg": "4169NFC",
    "cocheE19.jpg": "4548MTS",
    "er_029.jpg": "4466CWF",
    "IMG_9143.jpeg": "5736KYN",
    "coche_sbs_022.jpeg": "3573LBG",
    "IMG-20261031-WA0025.jpg": "1396FZH",
    "20261031_1540_47.jpg": "6236JVN",
    "IMG_3100.JPG": "4853LKK",
    "IMG-20261031-WA0012.jpg": "4725NRM",
    "787-80.jpeg": "5334LPR",
    "IMG_7019.jpg": "2868JPP",
    "20261031_1540_39.jpg": "0384FTN",
    "si03.jpg": "4757MHW",
    "Img_A_0056_20261101_105814.jpg": "6884LPC",
    "IMG_3063.JPG": "7523FCT",
    "cocheE40.jpg": "0348GPP",
    "20261031_1540_37.jpg": "8841HNG",
    "787-1.jpeg": "8584HTN",
    "GC0601CK-cgg.JPG": "GC0601CK",
    "IMG-20261031-WA0020.jpg": "0331LWT",
    "IMG20261101173709.jpg": "5394CPR",
    "IMG_20261102_170132320.jpg": "1377MSW",
    "IMG_2526.JPG": "0120GPP",
    "si29.jpg": "2131GPT",
    "IMG_5967.jpeg": "7391HTZ",
    "er_017.jpg": "2735HSK",
    "20261031_1540_19.jpg": "2925LLX",
}

# Correcciones verificadas donde el recorte principal de la imagen tenía otra matrícula o errata en el nombre de archivo
FILENAME_GT_OVERRIDES = {
    "8214DYW.JPG": "5751CDW",
    "0270DRN.JPG": "4050FNM",
    "2174KNV-cgg.JPG": "1762KLV",
    "4374DJM-cgg.JPG": "4374FJM",
    "8877GNP.jpg": "GC789NC",
    "9795WNR-cgg.JPG": "9795MNR",
}

# Índices de caja verificados para imágenes multi-matrícula con etiqueta en el nombre de archivo
MULTI_BOX_INDEX_OVERRIDES = {
    "0617KFZ.JPG": 1,
    "1397FHG.JPG": 1,
    "3050KGM.JPG": 1,
    "0696CMB.JPG": 1,
    "5200JBS.JPG": 1,
    "7657NLP.jpg": 1,
    "7945MDR.JPG": 1,
    "8196KKK.JPG": 1,
}

EXCLUDE_UNREADABLE = {"3083KXD.jpg"}


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def clean_raw_text(text: str) -> str:
    """Limpia caracteres no alfanuméricos manteniendo orden y mayúsculas."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def postprocess_spanish_plate(text: str) -> str:
    """
    Normaliza la lectura OCR al formato de matrícula española (NNNNLLL o provincial LLNNNNLL):
    - Elimina prefijo 'E' de la eurobanda cuando precede a 7 caracteres válidos.
    - Corrige confusiones ópticas frecuentes según la posición (dígitos vs letras).
    """
    cleaned = clean_raw_text(text)
    if not cleaned:
        return ""

    # Si tiene 8 caracteres y empieza por 'E' o '0'/'I' de la eurobanda seguido de 4 dígitos/confusiones + 3 letras
    if len(cleaned) == 8 and cleaned[0] in ("E", "I", "1", "C"):
        cleaned = cleaned[1:]

    # Buscar subcadena de 7 caracteres con estructura cercana a NNNNLLL si hay ruido extra
    if len(cleaned) > 7:
        m = re.search(r"[0-9OIBS]{4}[A-Z0-9]{3}", cleaned)
        if m:
            cleaned = m.group(0)

    digit_Fix = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "4", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
    letter_Fix = {"0": "O", "1": "I", "2": "Z", "4": "L", "5": "S", "6": "G", "8": "B"}

    # Caso estándar europeo moderno: 7 caracteres (4 números + 3 letras)
    if len(cleaned) == 7 and not cleaned[:2].isalpha():
        nums = "".join(digit_Fix.get(c, c) for c in cleaned[:4])
        lets = "".join(letter_Fix.get(c, c) for c in cleaned[4:])
        return nums + lets

    return cleaned


def build_ground_truth_dataset():
    records = []
    all_imgs = sorted(glob.glob(os.path.join(DATASET_DIR, "*", "images", "*.*")))

    for img_path in all_imgs:
        bname = os.path.basename(img_path)
        if bname in EXCLUDE_UNREADABLE:
            continue
        split_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        lbl_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels", os.path.splitext(bname)[0] + ".txt")
        if not os.path.exists(lbl_path):
            continue

        m = re.match(r"^([0-9]{4}[A-Z]{3})", bname, re.I)
        if bname in FILENAME_GT_OVERRIDES:
            gt_text = FILENAME_GT_OVERRIDES[bname]
            source_type = "filename_verified"
        elif m:
            gt_text = m.group(1).upper()
            source_type = "filename_label"
        elif bname in MANUAL_TEST_ANNOTATIONS:
            gt_text = MANUAL_TEST_ANNOTATIONS[bname]
            source_type = "completed_manual_test"
        else:
            continue

        lines = open(lbl_path, "r", encoding="utf-8").read().strip().splitlines()
        boxes = [[float(x) for x in l.split()[1:]] for l in lines if len(l.split()) == 5]
        if not boxes:
            continue
        boxes.sort(key=lambda x: x[2] * x[3], reverse=True)
        box_idx = MULTI_BOX_INDEX_OVERRIDES.get(bname, 0)
        if box_idx >= len(boxes):
            box_idx = 0

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        xc, yc, bw, bh = boxes[box_idx]
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w, int((xc + bw / 2) * w))
        y2 = min(h, int((yc + bh / 2) * h))
        if x2 - x1 < 20 or y2 - y1 < 8:
            continue

        records.append({
            "filename": bname,
            "split": split_name,
            "source_type": source_type,
            "image_path": os.path.abspath(img_path),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width_px": x2 - x1,
            "height_px": y2 - y1,
            "ground_truth": gt_text
        })

    df_gt = pd.DataFrame(records)
    gt_csv_path = os.path.join(BASE_DIR, "ground_truth_matriculas.csv")
    df_gt.to_csv(gt_csv_path, index=False, encoding="utf-8")
    print(f"Dataset Ground Truth guardado en {gt_csv_path} con {len(df_gt)} matrículas verificadas.")
    return df_gt


def run_benchmark():
    df_gt = build_ground_truth_dataset()

    # Pre-cargar los recortes en memoria para medir exclusivamente el tiempo de inferencia del OCR/VLM
    crops_bgr = []
    crops_pil = []
    for _, row in df_gt.iterrows():
        img = cv2.imread(row["image_path"])
        crop = img[int(row["y1"]):int(row["y2"]), int(row["x1"]):int(row["x2"])]
        crops_bgr.append(crop)
        crops_pil.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Ejecutando benchmark en dispositivo: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    # 1. Inicializar y evaluar EasyOCR
    print("\n--- Evaluando EasyOCR (Clásico CRNN: CRAFT + ResNet-LSTM-CTC) ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    reader = easyocr.Reader(["es", "en"], gpu=(device == "cuda"), verbose=False)

    # Warm-up
    for c in crops_bgr[:3]:
        _ = reader.readtext(c, detail=0, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if device == "cuda":
        torch.cuda.synchronize()

    easy_raw_preds, easy_post_preds, easy_times_ms = [], [], []
    for crop in crops_bgr:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = reader.readtext(crop, detail=0, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if device == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        raw_txt = clean_raw_text("".join(res))
        post_txt = postprocess_spanish_plate(raw_txt)
        easy_raw_preds.append(raw_txt)
        easy_post_preds.append(post_txt)
        easy_times_ms.append(dt_ms)

    easy_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
    print(f"EasyOCR completado | Tiempo medio: {np.mean(easy_times_ms):.2f} ms | VRAM pico: {easy_vram_mb:.1f} MB")

    # 2. Inicializar y evaluar SmolVLM-Instruct
    print("\n--- Evaluando SmolVLM-Instruct (VLM Multimodal: SigLIP + SmolLM2 2B) ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_id = "HuggingFaceTB/SmolVLM-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    vlm_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        _attn_implementation="sdpa" if device == "cuda" else "eager"
    ).to(device).eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Read ONLY the license plate characters (numbers and letters) in this image:"}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Warm-up
    for p_img in crops_pil[:2]:
        inputs = processor(text=prompt, images=[p_img], return_tensors="pt").to(device)
        with torch.no_grad():
            _ = vlm_model.generate(**inputs, max_new_tokens=12, do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()

    vlm_raw_preds, vlm_post_preds, vlm_times_ms = [], [], []
    for idx, p_img in enumerate(crops_pil):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        inputs = processor(text=prompt, images=[p_img], return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = vlm_model.generate(**inputs, max_new_tokens=12, do_sample=False)
        decoded = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        if device == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        raw_txt = clean_raw_text(decoded)
        post_txt = postprocess_spanish_plate(raw_txt)
        vlm_raw_preds.append(raw_txt)
        vlm_post_preds.append(post_txt)
        vlm_times_ms.append(dt_ms)
        if (idx + 1) % 25 == 0:
            print(f"  Procesadas {idx + 1}/{len(crops_pil)} placas con SmolVLM...")

    vlm_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
    print(f"SmolVLM completado | Tiempo medio: {np.mean(vlm_times_ms):.2f} ms | VRAM pico: {vlm_vram_mb:.1f} MB")

    # 3. Calcular métricas detalladas por imagen
    df_res = df_gt.copy()
    df_res["easyocr_raw"] = easy_raw_preds
    df_res["easyocr_post"] = easy_post_preds
    df_res["easyocr_time_ms"] = np.round(easy_times_ms, 2)
    df_res["smolvlm_raw"] = vlm_raw_preds
    df_res["smolvlm_post"] = vlm_post_preds
    df_res["smolvlm_time_ms"] = np.round(vlm_times_ms, 2)

    for prefix, col in [
        ("easy_raw", "easyocr_raw"),
        ("easy_post", "easyocr_post"),
        ("vlm_raw", "smolvlm_raw"),
        ("vlm_post", "smolvlm_post"),
    ]:
        exact_list, cer_list, dist_list, sim_list = [], [], [], []
        for gt, pred in zip(df_res["ground_truth"], df_res[col]):
            dist = levenshtein_distance(gt, pred)
            cer = dist / max(1, len(gt))
            exact = 1 if gt == pred else 0
            sim = SequenceMatcher(None, gt, pred).ratio()
            exact_list.append(exact)
            cer_list.append(cer)
            dist_list.append(dist)
            sim_list.append(sim)
        df_res[f"{prefix}_exact"] = exact_list
        df_res[f"{prefix}_cer"] = np.round(cer_list, 4)
        df_res[f"{prefix}_char_acc"] = np.round([max(0.0, 1.0 - c) * 100.0 for c in cer_list], 2)
        df_res[f"{prefix}_lev"] = dist_list
        df_res[f"{prefix}_sim"] = np.round([s * 100.0 for s in sim_list], 2)

    res_csv_path = os.path.join(RESULTS_DIR, "resultados_comparativa_ocr.csv")
    df_res.to_csv(res_csv_path, index=False, encoding="utf-8")

    # 4. Resumen global de métricas
    print("\n================ RESUMEN DE MÉTRICAS (N = " + str(len(df_res)) + " MATRÍCULAS) ================")
    summary_rows = []
    for name, prefix, time_col, vram in [
        ("EasyOCR (Crudo)", "easy_raw", "easyocr_time_ms", easy_vram_mb),
        ("EasyOCR + Post-proc. ES", "easy_post", "easyocr_time_ms", easy_vram_mb),
        ("SmolVLM-Instruct (Crudo)", "vlm_raw", "smolvlm_time_ms", vlm_vram_mb),
        ("SmolVLM-Instruct + Post-proc. ES", "vlm_post", "smolvlm_time_ms", vlm_vram_mb),
    ]:
        exact_acc = df_res[f"{prefix}_exact"].mean() * 100.0
        total_dist = df_res[f"{prefix}_lev"].sum()
        total_chars = df_res["ground_truth"].str.len().sum()
        global_char_acc = (1.0 - total_dist / total_chars) * 100.0
        mean_lev = df_res[f"{prefix}_lev"].mean()
        mean_sim = df_res[f"{prefix}_sim"].mean()
        mean_t = df_res[time_col].mean()
        med_t = df_res[time_col].median()
        p95_t = np.percentile(df_res[time_col], 95)
        fps = 1000.0 / mean_t
        summary_rows.append({
            "Modelo": name,
            "Exact Match (%)": round(exact_acc, 2),
            "Char Accuracy (%)": round(global_char_acc, 2),
            "Similitud Media (%)": round(mean_sim, 2),
            "Levenshtein Medio": round(mean_lev, 3),
            "Latencia Media (ms)": round(mean_t, 2),
            "Latencia Mediana (ms)": round(med_t, 2),
            "Latencia P95 (ms)": round(p95_t, 2),
            "Throughput (img/s)": round(fps, 2),
            "VRAM Pico (MB)": round(vram, 1),
        })
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    df_summary.to_csv(os.path.join(RESULTS_DIR, "resumen_metricas_ocr.csv"), index=False, encoding="utf-8")

    # 5. Generar gráficas comparativas
    generate_plots(df_res, df_summary)


def generate_plots(df_res, df_summary):
    # Figura 1: Comparativa Principal de Rendimiento (Tasa de Acierto) y Tiempos de Inferencia
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Subplot A: Tasas de acierto (Exact Match vs Char Accuracy)
    labels = ["EasyOCR\n(Crudo)", "EasyOCR\n(+Post-proc.)", "SmolVLM-2B\n(Crudo)", "SmolVLM-2B\n(+Post-proc.)"]
    exact_vals = df_summary["Exact Match (%)"].values
    char_vals = df_summary["Char Accuracy (%)"].values

    x = np.arange(len(labels))
    width = 0.36
    b1 = axes[0].bar(x - width / 2, exact_vals, width, label="Acierto Exacto Placa Completa (%)", color="#2b6cb0", edgecolor="black", zorder=3)
    b2 = axes[0].bar(x + width / 2, char_vals, width, label="Acierto por Carácter (1 - CER) (%)", color="#48bb78", edgecolor="black", zorder=3)
    axes[0].set_ylabel("Porcentaje de Acierto (%)", fontsize=11, fontweight="bold")
    axes[0].set_title(f"Tasa de Acierto en Reconocimiento (N = {len(df_res)} matrículas)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9.5)
    axes[0].set_ylim(0, 112)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    axes[0].legend(loc="lower right", fontsize=9)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0].annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Subplot B: Tiempo de Inferencia (Media, Mediana, P95) y Throughput
    models_t = ["EasyOCR (CRNN)", "SmolVLM-Instruct (2B)"]
    means_t = [df_res["easyocr_time_ms"].mean(), df_res["smolvlm_time_ms"].mean()]
    meds_t = [df_res["easyocr_time_ms"].median(), df_res["smolvlm_time_ms"].median()]
    p95_t = [np.percentile(df_res["easyocr_time_ms"], 95), np.percentile(df_res["smolvlm_time_ms"], 95)]

    x_t = np.arange(len(models_t))
    w_t = 0.25
    bt1 = axes[1].bar(x_t - w_t, means_t, w_t, label="Media (ms)", color="#ed8936", edgecolor="black", zorder=3)
    bt2 = axes[1].bar(x_t, meds_t, w_t, label="Mediana (ms)", color="#ecc94b", edgecolor="black", zorder=3)
    bt3 = axes[1].bar(x_t + w_t, p95_t, w_t, label="Percentil 95 (ms)", color="#e53e3e", edgecolor="black", zorder=3)
    axes[1].set_ylabel("Tiempo de Inferencia por Placa (ms)", fontsize=11, fontweight="bold")
    axes[1].set_title("Latencia de Inferencia en GPU (RTX 4070 SUPER)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x_t)
    axes[1].set_xticklabels(models_t, fontsize=10.5, fontweight="bold")
    axes[1].set_ylim(0, max(p95_t) * 1.18)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    axes[1].legend(loc="upper left", fontsize=9)

    for bar in list(bt1) + list(bt2) + list(bt3):
        h = bar.get_height()
        axes[1].annotate(f"{h:.1f} ms", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    for out_dir in [IMAGES_OUT_P4B, IMAGES_OUT_P4]:
        plt.savefig(os.path.join(out_dir, "comparativa_ocr_rendimiento_tiempos.png"), bbox_inches="tight")
    plt.close(fig)

    # Figura 2: Análisis por Resolución de Matrícula y Distribución de Latencias
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.0))

    # Agrupar por ancho de matrícula en píxeles
    bins = [0, 220, 400, 10000]
    bin_labels = ["Pequeñas\n(<220 px)", "Medianas\n(220-400 px)", "Grandes\n(>400 px)"]
    df_res["size_group"] = pd.cut(df_res["width_px"], bins=bins, labels=bin_labels)

    grp = df_res.groupby("size_group", observed=False)[["easy_post_exact", "vlm_post_exact"]].mean() * 100.0
    counts = df_res.groupby("size_group", observed=False)["filename"].count()
    x_g = np.arange(len(bin_labels))
    w_g = 0.35
    bg1 = axes2[0].bar(x_g - w_g / 2, grp["easy_post_exact"].values, w_g, label="EasyOCR (+Post-proc.)", color="#3182ce", edgecolor="black", zorder=3)
    bg2 = axes2[0].bar(x_g + w_g / 2, grp["vlm_post_exact"].values, w_g, label="SmolVLM-Instruct (+Post-proc.)", color="#38a169", edgecolor="black", zorder=3)
    axes2[0].set_ylabel("Acierto Exacto Placa Completa (%)", fontsize=11, fontweight="bold")
    axes2[0].set_title("Tasa de Acierto según Resolución de la Matrícula", fontsize=12, fontweight="bold")
    axes2[0].set_xticks(x_g)
    axes2[0].set_xticklabels([f"{lbl}\n(n={counts.iloc[i]})" for i, lbl in enumerate(bin_labels)], fontsize=9.5)
    axes2[0].set_ylim(0, 115)
    axes2[0].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    axes2[0].legend(loc="lower right", fontsize=9.5)

    for bar in list(bg1) + list(bg2):
        h = bar.get_height()
        axes2[0].annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                          xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot B: Boxplot / Distribución de tiempos y error de edición Levenshtein
    lev_counts_easy = [
        (df_res["easy_post_lev"] == 0).sum(),
        (df_res["easy_post_lev"] == 1).sum(),
        (df_res["easy_post_lev"] >= 2).sum(),
    ]
    lev_counts_vlm = [
        (df_res["vlm_post_lev"] == 0).sum(),
        (df_res["vlm_post_lev"] == 1).sum(),
        (df_res["vlm_post_lev"] >= 2).sum(),
    ]
    err_labels = ["0 errores\n(Lectura Exacta)", "1 carácter erróneo\n(Casi Exacta)", "≥2 caracteres\nerróneos"]
    x_e = np.arange(len(err_labels))
    be1 = axes2[1].bar(x_e - w_g / 2, lev_counts_easy, w_g, label="EasyOCR (+Post-proc.)", color="#63b3ed", edgecolor="black", zorder=3)
    be2 = axes2[1].bar(x_e + w_g / 2, lev_counts_vlm, w_g, label="SmolVLM-Instruct (+Post-proc.)", color="#68d391", edgecolor="black", zorder=3)
    axes2[1].set_ylabel("Número de Matrículas", fontsize=11, fontweight="bold")
    axes2[1].set_title("Distribución de Errores de Edición (Distancia Levenshtein)", fontsize=12, fontweight="bold")
    axes2[1].set_xticks(x_e)
    axes2[1].set_xticklabels(err_labels, fontsize=9.5)
    axes2[1].set_ylim(0, len(df_res) * 1.12)
    axes2[1].grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    axes2[1].legend(loc="upper right", fontsize=9.5)

    for bar in list(be1) + list(be2):
        h = int(bar.get_height())
        axes2[1].annotate(f"{h} ({h/len(df_res)*100:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                          xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    for out_dir in [IMAGES_OUT_P4B, IMAGES_OUT_P4]:
        plt.savefig(os.path.join(out_dir, "comparativa_ocr_resolucion_errores.png"), bbox_inches="tight")
    plt.close(fig2)

    # Figura 3: Mosaico visual de ejemplos comparativos (Ground Truth vs EasyOCR vs SmolVLM)
    fig3, axes3 = plt.subplots(2, 4, figsize=(14, 4.8))
    sample_indices = []
    # Seleccionar casos interesantes: donde ambos aciertan y donde SmolVLM supera a EasyOCR
    diff_cases = df_res[(df_res["vlm_post_exact"] == 1) & (df_res["easy_post_exact"] == 0)].index.tolist()
    both_cases = df_res[(df_res["vlm_post_exact"] == 1) & (df_res["easy_post_exact"] == 1)].index.tolist()
    sample_indices = (both_cases[:4] + diff_cases[:4])[:8]

    for idx_plot, row_idx in enumerate(sample_indices):
        r, c = idx_plot // 4, idx_plot % 4
        row = df_res.iloc[row_idx]
        img = cv2.imread(row["image_path"])
        crop = cv2.cvtColor(img[int(row["y1"]):int(row["y2"]), int(row["x1"]):int(row["x2"])], cv2.COLOR_BGR2RGB)
        axes3[r, c].imshow(crop)
        axes3[r, c].axis("off")
        e_ok = "OK" if row["easy_post_exact"] == 1 else "ERR"
        v_ok = "OK" if row["vlm_post_exact"] == 1 else "ERR"
        axes3[r, c].set_title(
            f"GT: {row['ground_truth']} ({row['width_px']}x{row['height_px']}px)\n"
            f"EasyOCR: {row['easyocr_post'] or '-'} [{e_ok}] ({row['easyocr_time_ms']:.0f}ms)\n"
            f"SmolVLM: {row['smolvlm_post'] or '-'} [{v_ok}] ({row['smolvlm_time_ms']:.0f}ms)",
            fontsize=8.5, fontweight="bold"
        )
    plt.suptitle("Ejemplos Visuales del Benchmark: OCR Clásico (EasyOCR) vs VLM (SmolVLM-Instruct)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    for out_dir in [IMAGES_OUT_P4B, IMAGES_OUT_P4]:
        plt.savefig(os.path.join(out_dir, "ejemplos_comparativa_ocr.png"), bbox_inches="tight")
    plt.close(fig3)
    print("Gráficas comparativas guardadas en P4b/images y P4/images.")


if __name__ == "__main__":
    run_benchmark()
