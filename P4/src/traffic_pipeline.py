import os
import re
import math
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter, defaultdict, deque
from pathlib import Path
from ultralytics import YOLO
try:
    from parking_identity import ParkingIdentity, parking_boxes
except ImportError:
    from src.parking_identity import ParkingIdentity, parking_boxes

COCO_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
COCO_PERSON_CLASS = 0
_HERE = Path(__file__).resolve().parent
_P4_ROOT = _HERE.parent
TRAFFIC_TRACKER = str(
    (_P4_ROOT / "botsort_trafico.yaml")
    if (_P4_ROOT / "botsort_trafico.yaml").exists()
    else (_HERE / "botsort_trafico.yaml")
)
PLATE_REFRESH_INTERVAL = 5


SPANISH_CONSONANTS = "BCDFGHJKLMNPRSTVWXYZ"


def clean_raw_plate_text(text: str) -> str:
    """Limpia caracteres no alfanuméricos manteniendo mayúsculas."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def postprocess_spanish_plate(text: str) -> str:
    """
    Normaliza la lectura OCR al formato de matrícula española (NNNNLLL o provincial LLNNNNLL):
    - Protege matrículas provinciales (ej. GC5273CM) antes de limpiar prefijos de eurobanda 'E'.
    - Corrige placas de 2 líneas invertidas (LLLNNNN -> NNNNLLL).
    - Corrige confusiones ópticas frecuentes según la posición (dígitos vs letras).
    """
    cleaned = clean_raw_plate_text(text)
    if not cleaned:
        return ""

    # Descartar alucinaciones repetitivas de 4+ caracteres iguales consecutivos (ej. 1000000, 55555)
    # sin rechazar matrículas válidas con tres dígitos iguales (ej. 2221FXK)
    if re.search(r"(.)\1{3}", cleaned):
        return ""

    digit_fix = {
        "O": "0", "Q": "0", "D": "0", "C": "0", "U": "0",
        "I": "1", "J": "1", "L": "4", "A": "4", "Z": "2", "S": "5",
        "B": "8", "G": "6", "T": "7"
    }
    letter_fix = {
        "0": "D", "1": "L", "2": "Z", "3": "B", "4": "L",
        "5": "S", "6": "G", "7": "Z", "8": "B", "9": "G",
        "O": "D", "I": "J", "Q": "G"
    }

    # Invertir lecturas de placa de 2 líneas leídas de abajo hacia arriba (LLLNNNN -> NNNNLLL)
    if re.match(r"^[A-Z]{3}\d{4}$", cleaned):
        cleaned = cleaned[3:] + cleaned[:3]

    # Detectar formato provincial (1-2 letras + 4 dígitos aprox + 1-2 letras) para no recortarlo como eurobanda
    is_provincial = bool(re.match(r"^[A-Z]{1,2}[0-9OQDCSBIZGT]{4}[A-Z0-9]{1,2}$", cleaned)) and not (
        len(cleaned) == 8 and cleaned[0] == "E" and re.match(r"^[0-9OQDCSBIZGT]{4}[A-Z0-9]{3}$", cleaned[1:])
    )

    if not is_provincial and len(cleaned) == 8 and cleaned[0] in ("E", "I", "1", "0", "L"):
        if re.match(r"^[0-9OQDCSBIZGT]{4}[A-Z0-9]{3}$", cleaned[1:]):
            cleaned = cleaned[1:]

    if not is_provincial and len(cleaned) > 7:
        m = re.search(r"[0-9OQDCSBIZGT]{4}[A-Z0-9]{3}", cleaned)
        if m:
            cleaned = m.group(0)

    if is_provincial:
        m_prov = re.match(r"^([A-Z]{1,2})([0-9OQDCSBIZGT]{4})([A-Z0-9]{1,2})$", cleaned)
        if m_prov:
            pref, nums, suf = m_prov.groups()
            nums = "".join(digit_fix.get(c, c) for c in nums)
            suf = "".join(letter_fix.get(c, c) for c in suf)
            return pref + nums + suf

    if len(cleaned) == 7 and not cleaned[:2].isalpha():
        nums = "".join(digit_fix.get(c, c) for c in cleaned[:4])
        lets = "".join(letter_fix.get(c, c) for c in cleaned[4:])
        return nums + lets

    return cleaned


def plate_text_quality_score(text: str) -> int:
    """
    Puntúa la validez estructural de una lectura de matrícula española y descarta alucinaciones.
    """
    if not text or len(text) < 6:
        return 0
    if len(set(text)) < 3 or re.search(r"(.)\1{3}", text):
        return 0
    if text in ("1234", "12345", "123456", "1234567", "3456", "2345"):
        return 0
    # Máxima prioridad: formato español estándar 4 dígitos + 3 consonantes (ej. 1965KBP)
    if re.match(rf"^\d{{4}}[{SPANISH_CONSONANTS}]{{3}}$", text):
        return 100
    # Formato provincial español (ej. GC5273CM, GC0601CK, M1234AB)
    if re.match(r"^[A-Z]{1,2}\d{4}[A-Z]{1,2}$", text):
        return 95
    # 4 dígitos + 3 letras cualesquiera
    if re.match(r"^\d{4}[A-Z]{3}$", text):
        return 92
    # Lectura parcial de 6-7 caracteres con al menos 3 dígitos y 2 letras
    n_digits = sum(c.isdigit() for c in text)
    n_alphas = sum(c.isalpha() for c in text)
    if n_digits >= 3 and n_alphas >= 2 and 6 <= len(text) <= 8:
        return 65 + len(text)
    return 0


def _vote_weight(score: int, eng_conf: float, pw: int, det_conf: float) -> float:
    """Peso de un voto OCR priorizando nitidez (confianza del motor) y resolución útil (<=140px)."""
    eff_pw = min(pw, 140)
    res_w = min(2.5, max(0.20, (eff_pw / 85.0) ** 2))
    return ((score / 100.0) ** 2) * (max(0.35, eng_conf) ** 2) * (0.7 + 0.3 * det_conf) * res_w


def _select_consensus_plate(votes):
    """
    Selecciona la lectura más fiable combinando múltiples fotogramas mediante votación ponderada
    (por calidad sintáctica, confianza del motor OCR, confianza de detección y resolución del recorte).
    Retorna (best_text, best_score, best_conf).
    """
    if not votes:
        return "", 0, 0.0

    str_weights = defaultdict(float)
    best_per_str = {}
    for text, sc, eng_conf, pw, det_conf in votes:
        if sc <= 0 or not text:
            continue
        w = _vote_weight(sc, eng_conf, pw, det_conf)
        str_weights[text] += w
        prev = best_per_str.get(text)
        if prev is None or (sc, eng_conf) > prev:
            best_per_str[text] = (sc, eng_conf)

    if not str_weights:
        return "", 0, 0.0

    best_str = max(str_weights, key=str_weights.get)
    best_sc, best_conf = best_per_str[best_str]
    target_len = len(best_str)

    same_len = [(t, sc, eng_conf, pw, det_conf) for (t, sc, eng_conf, pw, det_conf) in votes if len(t) == target_len and sc > 0]
    if len(same_len) >= 2 and target_len in (7, 8):
        pos_weights = [defaultdict(float) for _ in range(target_len)]
        for t, sc, eng_conf, pw, det_conf in same_len:
            w = _vote_weight(sc, eng_conf, pw, det_conf)
            for idx, ch in enumerate(t):
                pos_weights[idx][ch] += w

        chars = []
        for idx, cw in enumerate(pos_weights):
            if target_len == 7 and not best_str[:2].isalpha():
                allowed = [c for c in cw if (c.isdigit() if idx < 4 else c in SPANISH_CONSONANTS)]
                pool = allowed if allowed else list(cw.keys())
            else:
                pool = list(cw.keys())
            chars.append(max(pool, key=lambda c: cw[c]))

        cand = postprocess_spanish_plate("".join(chars))
        cand_sc = plate_text_quality_score(cand)
        if cand_sc >= best_sc and cand_sc >= 90:
            best_str = cand
            best_sc = cand_sc
            if cand in best_per_str:
                best_conf = max(best_conf, best_per_str[cand][1])

    return best_str, best_sc, best_conf


def associate_plate(plate_box, vehicles):
    """Choose the vehicle that contains most of the plate, favoring a centered plate."""
    x1, y1, x2, y2 = plate_box
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return None
    best_id, best_score = None, (0.0, 0.0)
    for vx1, vy1, vx2, vy2, track_id, _, _ in vehicles:
        common = max(0, min(x2, vx2) - max(x1, vx1)) * max(0, min(y2, vy2) - max(y1, vy1))
        coverage = common / area
        if coverage < 0.80:
            continue
        horizontal_offset = abs(x1 + x2 - vx1 - vx2) / max(1, vx2 - vx1)
        score = (coverage, -horizontal_offset)
        if score > best_score:
            best_id, best_score = track_id, score
    return best_id


def secondary_plate_position(plate, vehicle_box):
    """Return the relative position of a plausible plate from a secondary pass."""
    x1, y1, x2, y2 = plate[:4]
    vx1, vy1, vx2, vy2 = vehicle_box
    pw, ph = x2 - x1, y2 - y1
    vw, vh = vx2 - vx1, vy2 - vy1
    if vw <= 0 or vh <= 0 or ph < 2:
        return None
    rx = ((x1 + x2) / 2 - vx1) / vw
    ry = ((y1 + y2) / 2 - vy1) / vh
    if 5 <= pw <= vw * 0.55 and 1.25 <= pw / ph <= 6 and 0.12 <= rx <= 0.88 and 0.46 <= ry <= 0.95:
        return rx, ry
    return None


def detect_plate_contours(vehicle_bgr):
    """
    Detector clásico de matrículas mediante morfología matemática y contornos (Extra 3).
    Busca regiones rectangulares con contraste típico de placa en el tercio inferior del vehículo.
    """
    vh, vw = vehicle_bgr.shape[:2]
    if vh < 30 or vw < 40:
        return None

    y_start = int(vh * 0.45)
    roi = vehicle_bgr[y_start:, :]
    rh, rw = roi.shape[:2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    minVal, maxVal = np.min(gradX), np.max(gradX)
    if maxVal > minVal:
        gradX = ((gradX - minVal) / (maxVal - minVal) * 255).astype("uint8")
    else:
        gradX = gradX.astype("uint8")

    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    best_score = -1

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        ar = w / float(h)
        if 2.2 <= ar <= 6.0:
            area = w * h
            roi_area = rw * rh
            rel_area = area / roi_area
            if 0.01 <= rel_area <= 0.35:
                score = area * (1.0 - abs(ar - 4.5) / 4.5)
                if score > best_score:
                    best_score = score
                    best_candidate = (x, y + y_start, x + w, y + y_start + h)

    return best_candidate


def apply_anonymization(frame, boxes_to_blur):
    """
    Aplica desenfoque gaussiano de privacidad sobre las regiones seleccionadas (Extra 4).
    """
    h_img, w_img = frame.shape[:2]
    out = frame.copy()
    for (x1, y1, x2, y2) in boxes_to_blur:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
        bw = x2 - x1
        bh = y2 - y1
        if bw > 4 and bh > 4:
            kw = (bw // 2) * 2 + 1
            kh = (bh // 2) * 2 + 1
            kw = max(15, min(99, kw))
            kh = max(15, min(99, kh))
            roi = out[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (kw, kh), 30)
            out[y1:y2, x1:x2] = blurred_roi
    return out


class PlateOCR:
    """
    Módulo unificado de reconocimiento óptico de caracteres (OCR) para matrículas (Práctica 4b).
    Soporta:
      - 'fast_ocr': Modelo ONNX especializado en matrículas (fast-plate-ocr / cct-s-v2-global-model).
      - 'easyocr': OCR clásico basado en CRNN con ordenación de filas para placas de 2 líneas.
      - 'smolvlm': Modelo de Lenguaje Visual (HuggingFaceTB/SmolVLM-Instruct).
      - 'hybrid': Cascada rápida fast-plate-ocr -> EasyOCR -> SmolVLM en recortes ambiguos.
    """
    def __init__(self, mode="hybrid", device="cuda:0"):
        self.mode = mode.lower() if mode else "none"
        self.device = "cuda" if "cuda" in str(device) and torch.cuda.is_available() else "cpu"
        self.fast_reader = None
        self.easy_reader = None
        self.vlm_processor = None
        self.vlm_model = None
        self.vlm_prompt = None
        self._clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))

        if self.mode in ("hybrid", "fast_ocr"):
            try:
                from fast_plate_ocr import LicensePlateRecognizer
                print("Cargando lector especializado fast-plate-ocr (cct-s-v2-global-model) en cpu...")
                self.fast_reader = LicensePlateRecognizer("cct-s-v2-global-model", device="cpu")
            except Exception as exc:
                print(f"Aviso: no se pudo cargar fast-plate-ocr ({exc}); usando motores alternativos.")

        if self.mode in ("easyocr", "hybrid"):
            import easyocr
            print(f"Cargando lector OCR clásico EasyOCR en {self.device}...")
            self.easy_reader = easyocr.Reader(["es", "en"], gpu=(self.device == "cuda"), verbose=False)

        if self.mode in ("smolvlm", "hybrid"):
            from transformers import AutoProcessor, AutoModelForImageTextToText
            model_id = "HuggingFaceTB/SmolVLM-Instruct"
            print(f"Cargando modelo VLM '{model_id}' en {self.device} (bfloat16)...")
            self.vlm_processor = AutoProcessor.from_pretrained(model_id)
            self.vlm_model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                _attn_implementation="sdpa" if self.device == "cuda" else "eager"
            ).to(self.device).eval()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Read ONLY the license plate characters (numbers and letters) in this image:"}
                    ]
                }
            ]
            self.vlm_prompt = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=True)

    def _enhance_crop(self, crop_bgr, pad=True):
        """Mejora contraste local (CLAHE) y nitidez preservando bordes para OCR."""
        img = crop_bgr
        if pad:
            ph_pad = max(3, img.shape[0] // 8)
            pw_pad = max(4, img.shape[1] // 14)
            img = cv2.copyMakeBorder(img, ph_pad, ph_pad, pw_pad, pw_pad, cv2.BORDER_REPLICATE)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        enhanced = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        return cv2.addWeighted(enhanced, 1.45, blur, -0.45, 0)

    def _run_fast_ocr(self, crop_bgr):
        """Inferencia sobre recorte natural y variante realzada con fast-plate-ocr."""
        variants = [crop_bgr]
        if crop_bgr.shape[1] < 130:
            variants.append(self._enhance_crop(crop_bgr, pad=True))

        best = ("", 0, 0.0)
        for idx, var_bgr in enumerate(variants):
            rgb = cv2.cvtColor(var_bgr, cv2.COLOR_BGR2RGB)
            out = self.fast_reader.run(rgb, return_confidence=True)
            if not out:
                continue
            pred = out[0]
            raw = pred.plate.replace("_", "").strip().upper()
            if not raw:
                continue
            probs = pred.char_probs[:len(raw)]
            eng_conf = float(np.mean(probs)) if len(probs) > 0 else 0.0
            cand = postprocess_spanish_plate(raw)
            sc = plate_text_quality_score(cand)
            if (sc, eng_conf) > (best[1], best[2]):
                best = (cand, sc, eng_conf)
            if idx == 0 and sc == 100 and eng_conf >= 0.92:
                break
        return best

    def _run_easyocr(self, crop_bgr):
        """Inferencia EasyOCR ordenando cajas detectadas por filas (arriba->abajo, izq->der)."""
        enhanced = self._enhance_crop(crop_bgr, pad=True)
        up_bgr = cv2.resize(enhanced, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        detections = self.easy_reader.readtext(
            up_bgr, detail=1, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        if not detections:
            return "", 0, 0.0

        row_thresh = max(8.0, up_bgr.shape[0] * 0.28)
        sortable = []
        confs = []
        for bbox, txt, prob in detections:
            ys = [pt[1] for pt in bbox]
            xs = [pt[0] for pt in bbox]
            cy = sum(ys) / 4.0
            cx = sum(xs) / 4.0
            sortable.append((int(cy // row_thresh), cx, txt))
            confs.append(float(prob))
        sortable.sort(key=lambda item: (item[0], item[1]))
        joined = "".join(item[2] for item in sortable)
        cand = postprocess_spanish_plate(joined)
        sc = plate_text_quality_score(cand)
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return cand, sc, avg_conf

    def _run_smolvlm(self, crop_bgr):
        """Inferencia con SmolVLM-Instruct reescalando recortes pequeños para mejor legibilidad."""
        var_bgr = crop_bgr
        cw = var_bgr.shape[1]
        if cw < 96:
            scale = max(2.0, 180.0 / max(1, cw))
            var_bgr = cv2.resize(self._enhance_crop(var_bgr, pad=True), (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        pil_img = Image.fromarray(cv2.cvtColor(var_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.vlm_processor(text=self.vlm_prompt, images=[pil_img], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.vlm_model.generate(**inputs, max_new_tokens=12, do_sample=False)
        ans = self.vlm_processor.batch_decode(
            out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        cand = postprocess_spanish_plate(ans)
        sc = plate_text_quality_score(cand)
        return cand, sc, 0.85

    def read_plate_detail(self, crop_bgr, allow_vlm=True):
        """Devuelve (texto_matricula, calidad_sintactica, confianza_motor, used_vlm)."""
        if crop_bgr is None or crop_bgr.size == 0 or self.mode == "none":
            return "", 0, 0.0, False
        ch, cw = crop_bgr.shape[:2]
        if cw < 22 or ch < 9:
            return "", 0, 0.0, False

        candidates = []
        if self.fast_reader is not None:
            candidates.append(self._run_fast_ocr(crop_bgr))

        best = max(candidates, key=lambda item: (item[1], item[2]), default=("", 0, 0.0))
        if best[1] == 100 and best[2] >= 0.92:
            return best[0], best[1], best[2], False

        if self.easy_reader is not None and (best[1] < 100 or best[2] < 0.88):
            candidates.append(self._run_easyocr(crop_bgr))
            best = max(candidates, key=lambda item: (item[1], item[2]))

        used_vlm = False
        if allow_vlm and self.vlm_model is not None and cw >= 28 and ch >= 10 and (
            best[1] < 100 or best[2] < 0.90
        ):
            candidates.append(self._run_smolvlm(crop_bgr))
            used_vlm = True
            best = max(candidates, key=lambda item: (item[1], item[2]))

        # Si al menos dos motores distintos coinciden exactamente, reforzar confianza
        valid_texts = [t for t, sc, _ in candidates if sc >= 65 and t]
        if best[1] >= 65 and valid_texts.count(best[0]) >= 2:
            best = (best[0], best[1], max(best[2], 0.98))

        if best[1] < 65:
            return "", 0, 0.0, used_vlm
        return best[0], best[1], best[2], used_vlm

    def read_plate_with_conf(self, crop_bgr, allow_vlm=True):
        """Devuelve (texto_matricula, calidad_sintactica, confianza_motor)."""
        return self.read_plate_detail(crop_bgr, allow_vlm=allow_vlm)[:3]

    def read_plate(self, crop_bgr, allow_vlm=True) -> str:
        return self.read_plate_with_conf(crop_bgr, allow_vlm=allow_vlm)[0]


def _box_iou(box_a, box_b) -> float:
    """Calcula la intersección sobre unión (IoU) entre dos cajas (x1, y1, x2, y2)."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


class VehicleMotionFilter:
    """
    Filtro general de vehículos en movimiento basado en sustracción de fondo (MOG2)
    y análisis incremental de trayectoria con búfer lookahead.
    Distingue vehículos realmente desplazándose de coches aparcados o ruido estático.
    """

    def __init__(
        self,
        min_span_px=30.0,
        min_span_diag_ratio=0.55,
        max_start_iou=0.12,
        large_span_px=70.0,
        warmup_frames=25
    ):
        self.min_span_px = float(min_span_px)
        self.min_span_diag_ratio = float(min_span_diag_ratio)
        self.max_start_iou = float(max_start_iou)
        self.large_span_px = float(large_span_px)
        self.warmup_frames = int(warmup_frames)

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=25, detectShadows=False
        )
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._tracks = {}
        self.confirmed_moving = set()

    def _compute_fg_mask(self, frame_bgr):
        """Calcula máscara binaria de movimiento MOG2 reescalada a <=960px de ancho."""
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, 960.0 / max(1, w))
        if scale < 1.0:
            small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        else:
            small = frame_bgr
        fg = self.bg_subtractor.apply(small)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._morph_kernel)
        fg = cv2.dilate(fg, None, iterations=1)
        return fg, scale

    @staticmethod
    def _core_fg_ratio(fg_mask, scale, box) -> float:
        """Fracción de píxeles en movimiento en el núcleo central (60% interior) de la caja."""
        mh, mw = fg_mask.shape[:2]
        sx1 = max(0, int(box[0] * scale))
        sy1 = max(0, int(box[1] * scale))
        sx2 = min(mw, int(box[2] * scale))
        sy2 = min(mh, int(box[3] * scale))
        if sx2 <= sx1 or sy2 <= sy1:
            return 0.0
        px = max(1, (sx2 - sx1) // 5)
        py = max(1, (sy2 - sy1) // 5)
        cx1, cy1 = sx1 + px, sy1 + py
        cx2, cy2 = sx2 - px, sy2 - py
        if cx2 <= cx1 or cy2 <= cy1:
            roi = fg_mask[sy1:sy2, sx1:sx2]
        else:
            roi = fg_mask[cy1:cy2, cx1:cx2]
        return float((roi > 127).mean())

    def _is_moving_state(self, st) -> bool:
        """Evalúa si las métricas acumuladas de la trayectoria confirman movimiento real."""
        span = math.hypot(st["max_cx"] - st["min_cx"], st["max_cy"] - st["min_cy"])
        diag_med = float(np.median(st["diags"]))
        span_diag = span / max(1.0, diag_med)

        # Régimen 1: traslación consistente respecto al tamaño del vehículo saliendo de su huella inicial
        if (
            span >= self.min_span_px
            and span_diag >= self.min_span_diag_ratio
            and st["min_iou0"] <= self.max_start_iou
        ):
            return True

        # Régimen 2: gran desplazamiento absoluto para vehículos grandes cercanos
        if span >= self.large_span_px:
            return True

        # Régimen 3: evidencia combinada con primer plano real MOG2 en núcleo de caja
        fgs = st["core_fgs"]
        if fgs:
            if (
                len(fgs) >= 5
                and span >= max(35.0, self.min_span_px)
                and float(np.median(fgs)) >= 0.30
                and float(np.percentile(fgs, 75)) >= 0.58
            ):
                return True
            if (
                st["n_frames"] <= 8
                and span >= 12.0
                and diag_med >= 150.0
                and max(fgs) >= 0.65
            ):
                return True

        return False

    def update(self, frame_idx: int, frame_bgr, raw_vehicles):
        """Actualiza el modelo de fondo y las estadísticas de cada ID observado."""
        fg_mask, scale = self._compute_fg_mask(frame_bgr)
        fg_active = frame_idx > self.warmup_frames

        for (vx1, vy1, vx2, vy2, tid, _, _) in raw_vehicles:
            if tid in self.confirmed_moving:
                continue

            cx = (vx1 + vx2) * 0.5
            cy = (vy1 + vy2) * 0.5
            diag = math.hypot(vx2 - vx1, vy2 - vy1)
            box = (vx1, vy1, vx2, vy2)

            st = self._tracks.get(tid)
            if st is None:
                st = {
                    "first_box": box,
                    "min_cx": cx,
                    "max_cx": cx,
                    "min_cy": cy,
                    "max_cy": cy,
                    "diags": [diag],
                    "min_iou0": 1.0,
                    "core_fgs": [],
                    "n_frames": 0,
                }
                self._tracks[tid] = st

            st["n_frames"] += 1
            st["min_cx"] = min(st["min_cx"], cx)
            st["max_cx"] = max(st["max_cx"], cx)
            st["min_cy"] = min(st["min_cy"], cy)
            st["max_cy"] = max(st["max_cy"], cy)
            st["diags"].append(diag)
            st["min_iou0"] = min(st["min_iou0"], _box_iou(st["first_box"], box))
            if fg_active:
                st["core_fgs"].append(self._core_fg_ratio(fg_mask, scale, box))

            if self._is_moving_state(st):
                self.confirmed_moving.add(tid)

    def is_moving(self, track_id: int) -> bool:
        return track_id in self.confirmed_moving

    @property
    def filtered_count(self) -> int:
        return len(self._tracks) - len(self.confirmed_moving)


class TrafficAnalyzer:
    """
    Sistema integral de detección, tracking, análisis de flujo, localización de matrículas y OCR (P4 + P4b).
    """
    def __init__(
        self,
        general_model_path="yolo11m.pt",
        plate_model_path=None,
        conf_thres=0.10,
        imgsz=1280,
        device="cuda:0",
        ocr_mode="hybrid"
    ):
        print(f"Inicializando TrafficAnalyzer con {general_model_path} (imgsz={imgsz}) en {device}...")
        self.device = device
        self.conf_thres = conf_thres
        self.imgsz = imgsz

        self.general_model = YOLO(general_model_path)

        self.plate_model = YOLO(plate_model_path) if plate_model_path and os.path.exists(plate_model_path) else None
        if self.plate_model:
            print(f"Detector de matrículas cargado: {plate_model_path}")
        else:
            print("Detector de matrículas por Deep Learning no disponible aún. Usando método de contornos.")

        self.ocr = PlateOCR(mode=ocr_mode, device=device) if ocr_mode and ocr_mode != "none" else None

    def _detect_frame_plates(self, frame, vehicles, imgsz):
        results = self.plate_model(frame, conf=0.25, imgsz=imgsz, device=self.device, verbose=False)
        boxes = results[0].boxes
        plates = {}
        if boxes is not None:
            for pbox, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                coords = tuple(pbox.astype(int))
                track_id = associate_plate(coords, vehicles)
                if track_id is not None and (track_id not in plates or conf > plates[track_id][4]):
                    plates[track_id] = (*coords, float(conf), "")
        return plates

    def _detect_plate_crops(self, frame, vehicles):
        """Search small plates at a larger scale in the lower part of each vehicle."""
        crops, locations = [], []
        height, width = frame.shape[:2]
        for x1, y1, x2, y2, track_id, _, _ in vehicles:
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 - x1 < 28 or y2 - y1 < 22:
                continue
            crops.append(frame[y1:y2, x1:x2])
            locations.append((x1, y1, x2 - x1, y2 - y1, track_id))
        if not crops:
            return

        results = self.plate_model(crops, conf=0.30, imgsz=320, device=self.device, verbose=False)
        for (vx, vy, vw, vh, track_id), result in zip(locations, results):
            if result.boxes is None:
                continue
            best = None
            for pbox, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = pbox.astype(int)
                plate = (vx + x1, vy + y1, vx + x2, vy + y2, float(conf), "")
                position = secondary_plate_position(plate, (vx, vy, vx + vw, vy + vh))
                if position and x2 - x1 >= 6 and y2 - y1 >= 3 and (best is None or plate[4] > best[0][4]):
                    best = (plate, *position)
            if best is not None:
                yield track_id, *best

    @staticmethod
    def _update_directional_flow(state, cx, y2, track_id, width):
        """Actualiza trayectoria corta y métricas de flujo direccional por ID."""
        trail = state["track_history"][track_id]
        trail.append((cx, y2))
        if len(trail) > 30:
            trail.pop(0)
        if len(trail) >= 8:
            dx = trail[-1][0] - trail[0][0]
            if dx > 35:
                state["flow_left_to_right"].add(track_id)
            elif dx < -35:
                state["flow_right_to_left"].add(track_id)

        if track_id not in state["exited_ids"] and len(trail) >= 5:
            if cx > width - 60 and trail[-1][0] > trail[-3][0]:
                state["exits_right"] += 1
                state["exited_ids"].add(track_id)
            elif cx < 60 and trail[-1][0] < trail[-3][0]:
                state["exits_left"] += 1
                state["exited_ids"].add(track_id)

    def _recognize_frame_plates(
        self, state, frame, current_vehicles, width, height,
        native_plate_size, anonymize, use_contour_plates, frame_idx
    ):
        """Detecta matrículas en el fotograma (YOLO + crops + fallback contornos) y ejecuta OCR."""
        plates_in_frame = {}
        refresh_plates = frame_idx % PLATE_REFRESH_INTERVAL == 0
        needs_ocr_upgrade = (frame_idx % 2 == 0) and any(
            (
                110 <= v[2] - v[0] <= 920
                and (
                    state["vehicle_plate_scores"][v[4]] < 100
                    or state["vehicle_plate_confs"][v[4]] < 0.95
                    or state["vehicle_best_plate_w"][v[4]] < 120
                )
                and v[0] > 8
                and v[2] < width - 8
            )
            for v in current_vehicles
        )
        search_plates = (
            anonymize
            or (refresh_plates and bool(current_vehicles))
            or needs_ocr_upgrade
            or any(vehicle[4] not in state["unique_vehicles_with_plate"] for vehicle in current_vehicles)
        )

        if self.plate_model and search_plates:
            plates_in_frame = self._detect_frame_plates(frame, current_vehicles, self.imgsz)

        if self.plate_model and (refresh_plates or needs_ocr_upgrade):
            pending_vehicles = [
                vehicle for vehicle in current_vehicles
                if vehicle[4] not in plates_in_frame
                and (
                    vehicle[4] not in state["unique_vehicles_with_plate"]
                    or state["vehicle_plate_scores"][vehicle[4]] < 100
                    or state["vehicle_plate_confs"][vehicle[4]] < 0.95
                    or state["vehicle_best_plate_w"][vehicle[4]] < 120
                )
                and vehicle[6] != "bicycle"
            ]
            candidates = {v_id: (plate, rx, ry) for v_id, plate, rx, ry in self._detect_plate_crops(frame, pending_vehicles)}
            unmatched = {vehicle[4]: vehicle[:4] for vehicle in pending_vehicles if vehicle[4] not in candidates}
            if unmatched and native_plate_size > self.imgsz and refresh_plates:
                for v_id, plate in self._detect_frame_plates(frame, current_vehicles, native_plate_size).items():
                    if v_id in unmatched:
                        position = secondary_plate_position(plate, unmatched[v_id])
                        if position:
                            candidates[v_id] = (plate, *position)
            for v_id, (plate, rx, ry) in candidates.items():
                previous = state["secondary_observations"].get(v_id)
                if previous is not None:
                    last_frame, last_x, last_y = previous
                    recent = frame_idx - last_frame <= 3 * PLATE_REFRESH_INTERVAL
                    same_position = abs(rx - last_x) <= 0.12 and abs(ry - last_y) <= 0.12
                    if recent and same_position:
                        plates_in_frame[v_id] = plate
                elif plate[2] - plate[0] >= 28 and plate[4] >= 0.30:
                    plates_in_frame[v_id] = plate
                state["secondary_observations"][v_id] = (frame_idx, rx, ry)

        if use_contour_plates:
            for (vx1, vy1, vx2, vy2, v_id, _, _) in current_vehicles:
                if v_id not in plates_in_frame and (
                    anonymize or refresh_plates or v_id not in state["unique_vehicles_with_plate"]
                ):
                    vcrop = frame[vy1:vy2, vx1:vx2]
                    cbox = detect_plate_contours(vcrop)
                    if cbox:
                        cx1, cy1, cx2, cy2 = cbox
                        plates_in_frame[v_id] = (vx1 + cx1, vy1 + cy1, vx1 + cx2, vy1 + cy2, 0.65, "")

        if self.ocr is not None and not anonymize:
            for v_id, (mx1, my1, mx2, my2, mconf, mtext) in list(plates_in_frame.items()):
                pw, ph = mx2 - mx1, my2 - my1
                unclipped = (mx1 > 8 and mx2 < width - 8 and my1 > 8 and my2 < height - 8)
                votes = state["vehicle_plate_votes"][v_id]
                last_ocr_fr = state["vehicle_last_ocr_frame"].get(v_id, -999)
                sharp_votes = sum(1 for v in votes if v[3] >= 80 and v[2] >= 0.85)
                still_learning = (
                    state["vehicle_plate_scores"][v_id] < 100
                    or state["vehicle_plate_confs"][v_id] < 0.95
                    or sharp_votes < 4
                )
                grew_enough = (pw >= state["vehicle_best_plate_w"][v_id] * 1.15) and (pw <= 155)
                periodic_vote = (frame_idx - last_ocr_fr >= 3) and (36 <= pw <= 145) and still_learning

                if pw >= 24 and ph >= 10 and unclipped and (
                    state["vehicle_plate_scores"][v_id] == 0 or grew_enough or periodic_vote
                ):
                    pad_x, pad_y = max(3, pw // 14), max(2, ph // 8)
                    crop_bgr = frame[max(0, my1 - pad_y):min(height, my2 + pad_y),
                                     max(0, mx1 - pad_x):min(width, mx2 + pad_x)]
                    min_pw_vlm = 34 if mx1 > 1300 else 44
                    allow_vlm = (
                        state["vehicle_vlm_attempts"][v_id] < 4
                        and min_pw_vlm <= pw <= 480
                        and mconf >= 0.30
                    )
                    read_txt, sc, eng_conf, used_vlm = self.ocr.read_plate_detail(crop_bgr, allow_vlm=allow_vlm)
                    if used_vlm:
                        state["vehicle_vlm_attempts"][v_id] += 1
                    state["vehicle_last_ocr_frame"][v_id] = frame_idx
                    if sc > 0 and read_txt:
                        votes.append((read_txt, sc, eng_conf, pw, float(mconf)))
                        if len(votes) > 10:
                            votes.sort(key=lambda v: _vote_weight(v[1], v[2], v[3], v[4]), reverse=True)
                            del votes[10:]
                        best_txt, best_sc, best_conf = _select_consensus_plate(votes)
                        state["vehicle_plate_texts"][v_id] = best_txt
                        state["vehicle_plate_scores"][v_id] = best_sc
                        state["vehicle_plate_confs"][v_id] = best_conf
                    if sc > 0 or used_vlm:
                        state["vehicle_best_plate_w"][v_id] = max(state["vehicle_best_plate_w"][v_id], pw)

                plates_in_frame[v_id] = (mx1, my1, mx2, my2, mconf, state["vehicle_plate_texts"].get(v_id, mtext))

        return plates_in_frame

    @staticmethod
    def _append_frame_csv(state, frame_idx, current_persons, current_vehicles, plates_in_frame, new_plates):
        """Añade las detecciones de personas y vehículos del fotograma al registro CSV."""
        csv_records = state["csv_records"]
        first_plate_record_idx = state["first_plate_record_idx"]
        vehicle_plate_texts = state["vehicle_plate_texts"]

        for (px1, py1, px2, py2, p_id, p_conf, p_cls) in current_persons:
            csv_records.append({
                "fotograma": frame_idx,
                "tipo_objeto": p_cls,
                "confianza": round(float(p_conf), 3),
                "identificador_tracking": p_id,
                "x1": px1, "y1": py1, "x2": px2, "y2": py2,
                "matrícula_en_su_caso": "None",
                "confianza_matricula": "",
                "mx1": "", "my1": "", "mx2": "", "my2": "",
                "texto_matricula": ""
            })

        for (vx1, vy1, vx2, vy2, v_id, v_conf, v_cls) in current_vehicles:
            if v_id in new_plates:
                mx1, my1, mx2, my2, mconf, mtext = new_plates[v_id]
                has_plate = "Si"
                first_plate_record_idx[v_id] = len(csv_records)
            else:
                mx1, my1, mx2, my2, mconf, mtext = "", "", "", "", "", ""
                has_plate = "No"

            csv_records.append({
                "fotograma": frame_idx,
                "tipo_objeto": v_cls,
                "confianza": round(float(v_conf), 3),
                "identificador_tracking": v_id,
                "x1": vx1, "y1": vy1, "x2": vx2, "y2": vy2,
                "matrícula_en_su_caso": has_plate,
                "confianza_matricula": round(mconf, 3) if isinstance(mconf, float) else mconf,
                "mx1": mx1, "my1": my1, "mx2": mx2, "my2": my2,
                "texto_matricula": mtext
            })

            if v_id in first_plate_record_idx and v_id in vehicle_plate_texts:
                rec_idx = first_plate_record_idx[v_id]
                csv_records[rec_idx]["texto_matricula"] = vehicle_plate_texts[v_id]

    @staticmethod
    def _annotate_frame(
        state, annotated_frame, current_persons, current_vehicles, plates_in_frame,
        frame_idx, total_frames, width, anonymize
    ):
        """Dibuja cajas de personas, vehículos, matrículas proyectadas y HUD superior."""
        for (x1, y1, x2, y2, p_id, conf, _) in current_persons:
            color = (255, 128, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            lbl = f"Persona #{p_id} ({conf:.2f})"
            cv2.putText(annotated_frame, lbl, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        plate_offsets = state["plate_offsets"]
        vehicle_plate_texts = state["vehicle_plate_texts"]
        height = annotated_frame.shape[0]

        for (vx1, vy1, vx2, vy2, v_id, v_conf, v_cls) in current_vehicles:
            np.random.seed(v_id * 53 + 7)
            vcolor = tuple(int(c) for c in np.random.randint(50, 230, 3).tolist())
            cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), vcolor, 2)

            plate_txt = vehicle_plate_texts.get(v_id, "")
            v_lbl = f"{v_cls} #{v_id} ({v_conf:.2f})" + (f" [{plate_txt}]" if plate_txt and not anonymize else "")
            cv2.putText(annotated_frame, v_lbl, (vx1, max(20, vy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, vcolor, 2, cv2.LINE_AA)

            box_width = max(1, vx2 - vx1)
            box_height = max(1, vy2 - vy1)
            if v_id in plates_in_frame:
                mx1, my1, mx2, my2, _, _ = plates_in_frame[v_id]
                plate_offsets[v_id] = (
                    (mx1 - vx1) / box_width, (my1 - vy1) / box_height,
                    (mx2 - vx1) / box_width, (my2 - vy1) / box_height,
                )
            if v_id in plate_offsets:
                rx1, ry1, rx2, ry2 = plate_offsets[v_id]
                mx1 = max(0, round(vx1 + rx1 * box_width))
                my1 = max(0, round(vy1 + ry1 * box_height))
                mx2 = min(width - 1, round(vx1 + rx2 * box_width))
                my2 = min(height - 1, round(vy1 + ry2 * box_height))
                if mx1 < mx2 and my1 < my2:
                    cv2.rectangle(annotated_frame, (mx1, my1), (mx2, my2), (0, 255, 255), 2)
                    p_lbl = f"Matricula #{v_id}: {plate_txt}" if (plate_txt and not anonymize) else f"Matricula #{v_id}"
                    (tw, th), _ = cv2.getTextSize(p_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                    ty = max(18, my1 - 5)
                    cv2.rectangle(annotated_frame, (mx1, ty - th - 3), (min(width - 1, mx1 + tw + 4), ty + 3), (15, 15, 15), -1)
                    cv2.putText(annotated_frame, p_lbl, (mx1 + 2, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

        hud_bg = np.zeros((75, width, 3), dtype=np.uint8)
        cv2.rectangle(hud_bg, (0, 0), (width, 75), (20, 20, 20), -1)
        cv2.line(hud_bg, (0, 74), (width, 74), (0, 180, 255), 2)
        annotated_frame[0:75, 0:width] = cv2.addWeighted(annotated_frame[0:75, 0:width], 0.25, hud_bg, 0.75, 0)

        vehicle_breakdown = Counter(state["vehicle_classes"].values())
        state["vehicle_breakdown"] = vehicle_breakdown
        total_plates = len(state["unique_vehicles_with_plate"])
        line1 = (
            f"Fotograma: {frame_idx:04d}/{total_frames} | Personas Activas: {len(current_persons)} "
            f"(Total: {len(state['unique_persons'])}) | Vehiculos Activos: {len(current_vehicles)} "
            f"(Total: {len(state['vehicle_classes'])})"
        )
        veh_desglose = " | ".join(f"{kind.capitalize()}s: {count}" for kind, count in vehicle_breakdown.items())
        line2 = (
            f"Desglose: {veh_desglose if veh_desglose else 'None'} | Matriculas: {total_plates} "
            f"(OCR: {len(vehicle_plate_texts)}) | Flujo Dir: [Der->Izq: {len(state['flow_right_to_left'])}, "
            f"Izq->Der: {len(state['flow_left_to_right'])}]"
        )
        cv2.putText(annotated_frame, line1, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, line2, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)

    def _emit_buffered_frame(
        self, state, buffered_item, motion_filter, parking, width, height,
        native_plate_size, total_frames, anonymize, use_contour_plates, writer, display
    ):
        """Procesa y exporta un fotograma ya validado por el filtro de movimiento."""
        emit_idx, frame, raw_persons, raw_vehicles = buffered_item
        annotated_frame = frame.copy()
        boxes_to_blur = []

        current_persons = []
        for (x1, y1, x2, y2, track_id, conf, cname) in raw_persons:
            state["unique_persons"].add(track_id)
            current_persons.append((x1, y1, x2, y2, track_id, conf, cname))
            if anonymize:
                boxes_to_blur.append((x1, y1, x2, y2))
            if not (parking and parking.is_parked(track_id)):
                self._update_directional_flow(state, (x1 + x2) // 2, y2, track_id, width)

        current_vehicles = []
        for (x1, y1, x2, y2, track_id, conf, raw_cname) in raw_vehicles:
            if motion_filter is not None and not motion_filter.is_moving(track_id):
                continue
            votes = state["vehicle_class_votes"][track_id]
            votes[raw_cname] += 1
            cname = votes.most_common(1)[0][0]
            state["vehicle_classes"][track_id] = cname
            current_vehicles.append((x1, y1, x2, y2, track_id, conf, cname))
            if not (parking and parking.is_parked(track_id)):
                self._update_directional_flow(state, (x1 + x2) // 2, y2, track_id, width)

        plates_in_frame = self._recognize_frame_plates(
            state, frame, current_vehicles, width, height,
            native_plate_size, anonymize, use_contour_plates, emit_idx
        )
        new_plates = {
            v_id: plate for v_id, plate in plates_in_frame.items()
            if v_id not in state["unique_vehicles_with_plate"]
        }
        state["unique_vehicles_with_plate"].update(new_plates)
        if anonymize:
            for (mx1, my1, mx2, my2, _, _) in plates_in_frame.values():
                boxes_to_blur.append((mx1, my1, mx2, my2))

        self._append_frame_csv(state, emit_idx, current_persons, current_vehicles, plates_in_frame, new_plates)

        if anonymize and boxes_to_blur:
            annotated_frame = apply_anonymization(annotated_frame, boxes_to_blur)

        self._annotate_frame(
            state, annotated_frame, current_persons, current_vehicles,
            plates_in_frame, emit_idx, total_frames, width, anonymize
        )

        if writer:
            writer.write(annotated_frame)

        stop_requested = False
        if display:
            cv2.imshow("Sistema de Deteccion, Tracking y OCR (Grupo 33)", annotated_frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                print("Procesamiento detenido por el usuario.")
                stop_requested = True

        if emit_idx % 200 == 0:
            print(
                f"Fotograma {emit_idx}/{total_frames} | Personas: {len(state['unique_persons'])} | "
                f"Vehículos: {len(state['vehicle_classes'])} | Matrículas: {len(state['unique_vehicles_with_plate'])} | "
                f"Leídas con OCR: {len(state['vehicle_plate_texts'])}"
            )
        state["emitted_frames"] = emit_idx
        return stop_requested

    def process_video(
        self,
        video_path,
        output_video_path=None,
        output_csv_path=None,
        max_frames=None,
        anonymize=False,
        use_contour_plates=False,
        only_moving=True,
        motion_min_span=30.0,
        lookahead_frames=30,
        display=False
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el vídeo: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_plate_size = min(1920, max(width, height))
        anchors = parking_boxes(video_path, width, height)
        parking = ParkingIdentity(anchors) if anchors else None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames and max_frames < total_frames:
            total_frames = max_frames

        writer = None
        if output_video_path:
            out_v_dir = os.path.dirname(os.path.abspath(output_video_path))
            if out_v_dir:
                os.makedirs(out_v_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        motion_filter = VehicleMotionFilter(min_span_px=motion_min_span) if only_moving else None
        effective_lookahead = max(0, int(lookahead_frames)) if motion_filter is not None else 0
        pending_frames = deque()

        state = {
            "track_history": defaultdict(list),
            "unique_persons": set(),
            "vehicle_class_votes": defaultdict(Counter),
            "vehicle_classes": {},
            "vehicle_breakdown": Counter(),
            "unique_vehicles_with_plate": set(),
            "secondary_observations": {},
            "plate_offsets": {},
            "vehicle_plate_texts": {},
            "vehicle_plate_scores": defaultdict(int),
            "vehicle_plate_confs": defaultdict(float),
            "vehicle_plate_votes": defaultdict(list),
            "vehicle_last_ocr_frame": {},
            "vehicle_vlm_attempts": defaultdict(int),
            "vehicle_best_plate_w": defaultdict(int),
            "flow_left_to_right": set(),
            "flow_right_to_left": set(),
            "exits_left": 0,
            "exits_right": 0,
            "exited_ids": set(),
            "csv_records": [],
            "first_plate_record_idx": {},
            "emitted_frames": 0,
        }

        mode_desc = f"solo en movimiento (umbral={motion_min_span:.0f}px, lookahead={effective_lookahead}f)" if only_moving else "todos (incluye aparcados)"
        print(f"\nIniciando procesamiento de '{os.path.basename(video_path)}' [{mode_desc}]")
        print(f"Resolución: {width}x{height} | FPS: {fps:.2f} | Fotogramas a procesar: {total_frames}")

        frame_idx = 0
        stopped_early = False
        classes_to_track = [COCO_PERSON_CLASS] + list(COCO_VEHICLE_CLASSES.keys())

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            frame_idx += 1
            track_results = self.general_model.track(
                frame,
                persist=True,
                classes=classes_to_track,
                conf=self.conf_thres,
                imgsz=self.imgsz,
                tracker=TRAFFIC_TRACKER,
                device=self.device,
                verbose=False
            )

            raw_persons = []
            raw_vehicles = []
            if track_results[0].boxes is not None and track_results[0].boxes.id is not None:
                boxes = track_results[0].boxes.xyxy.cpu().numpy()
                track_ids = track_results[0].boxes.id.int().cpu().tolist()
                confidences = track_results[0].boxes.conf.cpu().numpy()
                clses = track_results[0].boxes.cls.int().cpu().tolist()

                detections = zip(boxes, track_ids, confidences, clses)
                if parking:
                    detections = parking.relabel(detections, frame_idx)
                for box, track_id, conf, cls_id in detections:
                    x1, y1, x2, y2 = box.astype(int)
                    if cls_id == COCO_PERSON_CLASS:
                        raw_persons.append((x1, y1, x2, y2, track_id, conf, "person"))
                    elif cls_id in COCO_VEHICLE_CLASSES:
                        raw_vehicles.append((x1, y1, x2, y2, track_id, conf, COCO_VEHICLE_CLASSES[cls_id]))

            if motion_filter is not None:
                motion_filter.update(frame_idx, frame, raw_vehicles)

            pending_frames.append((frame_idx, frame, raw_persons, raw_vehicles))
            while len(pending_frames) > effective_lookahead:
                stopped_early = self._emit_buffered_frame(
                    state, pending_frames.popleft(), motion_filter, parking,
                    width, height, native_plate_size, total_frames,
                    anonymize, use_contour_plates, writer, display
                )
                if stopped_early:
                    break
            if stopped_early:
                break

        while not stopped_early and pending_frames:
            stopped_early = self._emit_buffered_frame(
                state, pending_frames.popleft(), motion_filter, parking,
                width, height, native_plate_size, total_frames,
                anonymize, use_contour_plates, writer, display
            )

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        df = pd.DataFrame(state["csv_records"])
        if output_csv_path:
            out_c_dir = os.path.dirname(os.path.abspath(output_csv_path))
            if out_c_dir:
                os.makedirs(out_c_dir, exist_ok=True)
            column_names = [
                "fotograma", "tipo_objeto", "confianza", "identificador_tracking",
                "x1", "y1", "x2", "y2", "matrícula_en_su_caso", "confianza",
                "mx1", "my1", "mx2", "my2", "texto_matricula"
            ]
            df.to_csv(output_csv_path, index=False, header=column_names, encoding="utf-8")
            print(f"\nArchivo CSV guardado en: {output_csv_path}")

        vehicle_classes = state["vehicle_classes"]
        vehicle_breakdown = Counter(vehicle_classes.values())
        unique_vehicles_with_plate = state["unique_vehicles_with_plate"]
        vehicle_plate_texts = state["vehicle_plate_texts"]
        filtered_static = motion_filter.filtered_count if motion_filter else 0

        print("\n=== RESUMEN FINAL DEL PROCESAMIENTO ===")
        print(f"Total fotogramas procesados: {state['emitted_frames']}")
        print(f"Total personas únicas detectadas: {len(state['unique_persons'])}")
        print(f"Total vehículos únicos detectados: {len(vehicle_classes)}")
        if motion_filter is not None:
            print(f"Trayectorias estacionarias/aparcadas filtradas: {filtered_static}")
        for kind, count in vehicle_breakdown.items():
            print(f"  - {kind}: {count}")
        print(f"Flujo izquierda a derecha: {len(state['flow_left_to_right'])}")
        print(f"Flujo derecha a izquierda: {len(state['flow_right_to_left'])}")
        print(f"Vehículos que salieron por la derecha: {state['exits_right']}")
        print(f"Vehículos que salieron por la izquierda: {state['exits_left']}")
        print(f"Total matrículas registradas: {len(unique_vehicles_with_plate)}")
        print(f"Matrículas visibles con texto reconocido por OCR: {len(vehicle_plate_texts)}")
        if vehicle_plate_texts:
            print("Listado de matrículas leídas (ID -> Texto):")
            for vid, ptxt in sorted(vehicle_plate_texts.items()):
                print(f"  - Vehículo #{vid}: {ptxt}")
        if output_video_path:
            print(f"Vídeo de salida exportado en: {output_video_path}")

        return {
            "df": df,
            "unique_persons": len(state["unique_persons"]),
            "unique_vehicles": len(vehicle_classes),
            "filtered_stationary_tracks": filtered_static,
            "unique_vehicles_with_plate": len(unique_vehicles_with_plate),
            "recognized_plates_count": len(vehicle_plate_texts),
            "recognized_plates": dict(vehicle_plate_texts),
            "vehicle_breakdown": dict(vehicle_breakdown),
            "flow_l2r": len(state["flow_left_to_right"]),
            "flow_r2l": len(state["flow_right_to_left"]),
            "exits_right": state["exits_right"],
            "exits_left": state["exits_left"],
            "total_plates": len(unique_vehicles_with_plate)
        }
