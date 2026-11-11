import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter, defaultdict
from pathlib import Path
from ultralytics import YOLO
from parking_identity import ParkingIdentity, parking_boxes

COCO_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
COCO_PERSON_CLASS = 0
TRAFFIC_TRACKER = str(Path(__file__).with_name("botsort_trafico.yaml"))
PLATE_REFRESH_INTERVAL = 5


def clean_raw_plate_text(text: str) -> str:
    """Limpia caracteres no alfanuméricos manteniendo mayúsculas."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def postprocess_spanish_plate(text: str) -> str:
    """
    Normaliza la lectura OCR al formato de matrícula española (NNNNLLL o provincial LLNNNNLL):
    - Elimina prefijo 'E' de la eurobanda cuando precede a 7 caracteres válidos.
    - Corrige confusiones ópticas frecuentes según la posición (dígitos vs consonantes).
    """
    cleaned = clean_raw_plate_text(text)
    if not cleaned:
        return ""

    # Descartar alucinaciones numéricas repetitivas (ej. 1000000, 55555)
    if re.search(r"(.)\1\1", cleaned):
        return ""

    if len(cleaned) == 8 and cleaned[0] in ("E", "I", "1", "C", "0"):
        cleaned = cleaned[1:]

    if len(cleaned) > 7:
        m = re.search(r"[0-9OIBSC]{4}[A-Z0-9]{3}", cleaned)
        if m:
            cleaned = m.group(0)

    digit_fix = {
        "O": "0", "Q": "0", "D": "0", "C": "0",
        "I": "1", "L": "4", "Z": "2", "S": "5",
        "B": "8", "G": "6", "T": "7"
    }
    letter_fix = {
        "0": "D", "1": "L", "2": "Z", "4": "L",
        "5": "S", "6": "G", "8": "B", "O": "D", "I": "J"
    }

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
    if len(set(text)) < 4 or re.search(r"(.)\1\1", text):
        return 0
    if text in ("1234", "12345", "123456", "1234567", "3456", "2345"):
        return 0
    # Máxima prioridad: formato español estándar 4 dígitos + 3 consonantes (ej. 1965KBP)
    if re.match(r"^\d{4}[BCDFGHJKLMNPRSTVWXYZ]{3}$", text):
        return 100
    # 4 dígitos + 3 letras cualesquiera
    if re.match(r"^\d{4}[A-Z]{3}$", text):
        return 92
    # Formato provincial español (ej. GC1808CK, GP5273CM, M1234AB)
    if re.match(r"^[A-Z]{1,2}\d{4}[A-Z]{1,2}$", text):
        return 90
    # Lectura parcial de 6-7 caracteres con al menos 3 dígitos y 2 letras
    n_digits = sum(c.isdigit() for c in text)
    n_alphas = sum(c.isalpha() for c in text)
    if n_digits >= 3 and n_alphas >= 2 and 6 <= len(text) <= 8:
        return 65 + len(text)
    return 0


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
      - 'easyocr': OCR clásico basado en CRNN (CRAFT + ResNet-LSTM-CTC).
      - 'smolvlm': Modelo de Lenguaje Visual (HuggingFaceTB/SmolVLM-Instruct).
      - 'hybrid': Combina rapidez de EasyOCR con la alta tasa de acierto de SmolVLM en recortes nítidos.
    """
    def __init__(self, mode="hybrid", device="cuda:0"):
        self.mode = mode.lower() if mode else "none"
        self.device = "cuda" if "cuda" in str(device) and torch.cuda.is_available() else "cpu"
        self.easy_reader = None
        self.vlm_processor = None
        self.vlm_model = None
        self.vlm_prompt = None

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

    def read_plate(self, crop_bgr, allow_vlm=True) -> str:
        if crop_bgr is None or crop_bgr.size == 0 or self.mode == "none":
            return ""
        ch, cw = crop_bgr.shape[:2]
        if cw < 26 or ch < 9:
            return ""

        best_text = ""
        best_score = 0

        # 1. Intento rápido con EasyOCR sobre recorte reescalado bicúbico
        if self.easy_reader is not None:
            up_bgr = cv2.resize(crop_bgr, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            res = self.easy_reader.readtext(up_bgr, detail=0, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            cand = postprocess_spanish_plate("".join(res))
            sc = plate_text_quality_score(cand)
            if sc > best_score:
                best_text, best_score = cand, sc

        # 2. Si no se obtuvo matrícula española completa (score < 100) y SmolVLM está habilitado
        if best_score < 100 and allow_vlm and self.vlm_model is not None and cw >= 32 and ch >= 11:
            pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            inputs = self.vlm_processor(text=self.vlm_prompt, images=[pil_img], return_tensors="pt").to(self.device)
            with torch.no_grad():
                out_ids = self.vlm_model.generate(**inputs, max_new_tokens=12, do_sample=False)
            ans = self.vlm_processor.batch_decode(
                out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
            cand_vlm = postprocess_spanish_plate(ans)
            sc_vlm = plate_text_quality_score(cand_vlm)
            if sc_vlm >= best_score and sc_vlm > 0:
                best_text, best_score = cand_vlm, sc_vlm

        return best_text if best_score >= 60 else ""


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

    def process_video(
        self,
        video_path,
        output_video_path=None,
        output_csv_path=None,
        max_frames=None,
        anonymize=False,
        use_contour_plates=False,
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
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Estructuras de seguimiento y métricas
        track_history = defaultdict(lambda: [])
        unique_persons = set()
        vehicle_class_votes = defaultdict(Counter)
        vehicle_classes = {}
        vehicle_breakdown = Counter()
        unique_vehicles_with_plate = set()
        secondary_observations = {}
        plate_offsets = {}  # Coordenadas normalizadas respecto al recuadro de cada vehículo.

        # Estructuras de reconocimiento OCR por vehículo (P4b)
        vehicle_plate_texts = {}       # v_id -> mejor texto reconocido (ej. '1965KBP')
        vehicle_plate_scores = defaultdict(int)  # v_id -> calidad de lectura (0..100)
        vehicle_vlm_attempts = defaultdict(int)  # v_id -> nº de llamadas VLM realizadas
        vehicle_best_plate_w = defaultdict(int)  # v_id -> mayor ancho de placa evaluado con OCR

        # Estadísticas de flujo direccional (Extra 1)
        flow_left_to_right = set()
        flow_right_to_left = set()
        exits_left = 0
        exits_right = 0
        exited_ids = set()

        # Registros para el archivo CSV oficial
        csv_records = []
        first_plate_record_idx = {}  # v_id -> índice en csv_records donde se registró 'Si'

        print(f"\nIniciando procesamiento de '{os.path.basename(video_path)}'")
        print(f"Resolución: {width}x{height} | FPS: {fps:.2f} | Fotogramas a procesar: {total_frames}")

        frame_idx = 0
        classes_to_track = [COCO_PERSON_CLASS] + list(COCO_VEHICLE_CLASSES.keys())

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            frame_idx += 1
            annotated_frame = frame.copy()
            boxes_to_blur = []

            # 1. Inferencia de Detección y Seguimiento Persistente con YOLO
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

            current_vehicles = []  # [(vx1, vy1, vx2, vy2, track_id, conf, class_name), ...]
            current_persons = []

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
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if cls_id == COCO_PERSON_CLASS:
                        unique_persons.add(track_id)
                        current_persons.append((x1, y1, x2, y2, track_id, conf, "person"))
                        if anonymize:
                            boxes_to_blur.append((x1, y1, x2, y2))
                    elif cls_id in COCO_VEHICLE_CLASSES:
                        cname = COCO_VEHICLE_CLASSES[cls_id]
                        votes = vehicle_class_votes[track_id]
                        votes[cname] += 1
                        cname = votes.most_common(1)[0][0]
                        vehicle_classes[track_id] = cname
                        current_vehicles.append((x1, y1, x2, y2, track_id, conf, cname))

                    # 2. Análisis de Flujo Direccional (Extra 1)
                    if parking and parking.is_parked(track_id):
                        continue
                    trail = track_history[track_id]
                    trail.append((cx, y2))
                    if len(trail) > 30:
                        trail.pop(0)
                    if len(trail) >= 8:
                        dx = trail[-1][0] - trail[0][0]
                        if dx > 35:
                            flow_left_to_right.add(track_id)
                        elif dx < -35:
                            flow_right_to_left.add(track_id)

                    if track_id not in exited_ids and len(trail) >= 5:
                        if cx > width - 60 and trail[-1][0] > trail[-3][0]:
                            exits_right += 1
                            exited_ids.add(track_id)
                        elif cx < 60 and trail[-1][0] < trail[-3][0]:
                            exits_left += 1
                            exited_ids.add(track_id)

            # 3. Buscar placas nuevas y refrescar periódicamente las que ya tienen un ID
            # También buscamos si un vehículo está en primer plano (vw >= 150) y aún no tiene lectura OCR completa
            plates_in_frame = {}  # vehicle_track_id -> (mx1, my1, mx2, my2, conf_m, texto_m)
            refresh_plates = frame_idx % PLATE_REFRESH_INTERVAL == 0
            needs_ocr_upgrade = (frame_idx % 2 == 0) and any(
                (140 <= v[2] - v[0] <= 680 and vehicle_plate_scores[v[4]] < 100 and v[0] > 15 and v[2] < width - 15)
                for v in current_vehicles
            )
            search_plates = (
                anonymize
                or (refresh_plates and bool(current_vehicles))
                or needs_ocr_upgrade
                or any(vehicle[4] not in unique_vehicles_with_plate for vehicle in current_vehicles)
            )

            if self.plate_model and search_plates:
                plates_in_frame = self._detect_frame_plates(frame, current_vehicles, self.imgsz)

            if self.plate_model and (refresh_plates or needs_ocr_upgrade):
                pending_vehicles = [
                    vehicle for vehicle in current_vehicles
                    if vehicle[4] not in plates_in_frame
                    and (vehicle[4] not in unique_vehicles_with_plate or vehicle_plate_scores[vehicle[4]] < 100)
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
                    previous = secondary_observations.get(v_id)
                    if previous is not None:
                        last_frame, last_x, last_y = previous
                        recent = frame_idx - last_frame <= 3 * PLATE_REFRESH_INTERVAL
                        same_position = abs(rx - last_x) <= 0.12 and abs(ry - last_y) <= 0.12
                        if recent and same_position:
                            plates_in_frame[v_id] = plate
                    elif plate[2] - plate[0] >= 28 and plate[4] >= 0.30:
                        plates_in_frame[v_id] = plate
                    secondary_observations[v_id] = (frame_idx, rx, ry)

            if use_contour_plates:
                for (vx1, vy1, vx2, vy2, v_id, _, _) in current_vehicles:
                    if v_id not in plates_in_frame and (anonymize or refresh_plates or v_id not in unique_vehicles_with_plate):
                        vcrop = frame[vy1:vy2, vx1:vx2]
                        cbox = detect_plate_contours(vcrop)
                        if cbox:
                            cx1, cy1, cx2, cy2 = cbox
                            plates_in_frame[v_id] = (vx1 + cx1, vy1 + cy1, vx1 + cx2, vy1 + cy2, 0.65, "")

            # 3b. Reconocimiento OCR sobre las matrículas visibles en el fotograma (Práctica 4b)
            if self.ocr is not None and not anonymize:
                for v_id, (mx1, my1, mx2, my2, mconf, mtext) in list(plates_in_frame.items()):
                    pw, ph = mx2 - mx1, my2 - my1
                    unclipped = (mx1 > 15 and mx2 < width - 15 and my1 > 12 and my2 < height - 12)
                    min_pw_vlm = 42 if mx1 > 1350 else 58
                    # Ejecutar OCR cuando la placa está en zona nítida y ofrece mayor resolución que lecturas previas
                    if pw >= 36 and ph >= 12 and unclipped and (vehicle_plate_scores[v_id] < 100 or vehicle_best_plate_w[v_id] < 78):
                        if pw >= vehicle_best_plate_w[v_id] * 1.20 or vehicle_plate_scores[v_id] == 0:
                            pad_x, pad_y = max(2, pw // 18), max(2, ph // 10)
                            crop_bgr = frame[max(0, my1 - pad_y):min(height, my2 + pad_y),
                                             max(0, mx1 - pad_x):min(width, mx2 + pad_x)]
                            allow_vlm = (
                                vehicle_vlm_attempts[v_id] < 4
                                and min_pw_vlm <= pw <= 148
                                and mconf >= 0.35
                            )
                            read_txt = self.ocr.read_plate(crop_bgr, allow_vlm=allow_vlm)
                            if allow_vlm:
                                vehicle_vlm_attempts[v_id] += 1
                            sc = plate_text_quality_score(read_txt)
                            if sc > vehicle_plate_scores[v_id] or (sc == vehicle_plate_scores[v_id] and sc >= 90 and 65 <= pw <= 135):
                                vehicle_plate_scores[v_id] = sc
                                vehicle_plate_texts[v_id] = read_txt
                            if sc > 0 or allow_vlm:
                                vehicle_best_plate_w[v_id] = max(vehicle_best_plate_w[v_id], pw)

                    # Asociar el mejor texto conocido del vehículo a la detección actual
                    plates_in_frame[v_id] = (mx1, my1, mx2, my2, mconf, vehicle_plate_texts.get(v_id, mtext))

            new_plates = {
                v_id: plate for v_id, plate in plates_in_frame.items()
                if v_id not in unique_vehicles_with_plate
            }
            unique_vehicles_with_plate.update(new_plates)
            total_plates_detected = len(unique_vehicles_with_plate)
            if anonymize:
                for (mx1, my1, mx2, my2, _, _) in plates_in_frame.values():
                    boxes_to_blur.append((mx1, my1, mx2, my2))

            # 4. Registrar filas para el CSV Oficial
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

                # Si en un fotograma posterior más cercano se logra leer el texto o mejorar la caja de la matrícula,
                # actualizamos el registro de matrícula del vehículo en el CSV con el texto reconocido.
                if v_id in first_plate_record_idx and v_id in vehicle_plate_texts:
                    rec_idx = first_plate_record_idx[v_id]
                    csv_records[rec_idx]["texto_matricula"] = vehicle_plate_texts[v_id]
                    if v_id in plates_in_frame:
                        pmx1, pmy1, pmx2, pmy2, pmconf, _ = plates_in_frame[v_id]
                        if pmconf > float(csv_records[rec_idx]["confianza_matricula"] or 0.0):
                            csv_records[rec_idx]["mx1"] = pmx1
                            csv_records[rec_idx]["my1"] = pmy1
                            csv_records[rec_idx]["mx2"] = pmx2
                            csv_records[rec_idx]["my2"] = pmy2
                            csv_records[rec_idx]["confianza_matricula"] = round(float(pmconf), 3)

            # 5. Aplicar anonimización de privacidad si está activa (Extra 4)
            if anonymize and boxes_to_blur:
                annotated_frame = apply_anonymization(annotated_frame, boxes_to_blur)

            # 6. Renderizado de anotaciones visuales
            for (x1, y1, x2, y2, p_id, conf, _) in current_persons:
                color = (255, 128, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                lbl = f"Persona #{p_id} ({conf:.2f})"
                cv2.putText(annotated_frame, lbl, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

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

            # 7. Renderizado del HUD Superior Profesional
            hud_bg = np.zeros((75, width, 3), dtype=np.uint8)
            cv2.rectangle(hud_bg, (0, 0), (width, 75), (20, 20, 20), -1)
            cv2.line(hud_bg, (0, 74), (width, 74), (0, 180, 255), 2)
            annotated_frame[0:75, 0:width] = cv2.addWeighted(annotated_frame[0:75, 0:width], 0.25, hud_bg, 0.75, 0)

            vehicle_breakdown = Counter(vehicle_classes.values())
            line1 = f"Fotograma: {frame_idx:04d}/{total_frames} | Personas Activas: {len(current_persons)} (Total: {len(unique_persons)}) | Vehiculos Activos: {len(current_vehicles)} (Total: {len(vehicle_classes)})"
            veh_desglose = " | ".join(f"{kind.capitalize()}s: {count}" for kind, count in vehicle_breakdown.items())
            line2 = f"Desglose: {veh_desglose if veh_desglose else 'None'} | Matriculas: {total_plates_detected} (OCR: {len(vehicle_plate_texts)}) | Flujo Dir: [Der->Izq: {len(flow_right_to_left)}, Izq->Der: {len(flow_left_to_right)}]"

            cv2.putText(annotated_frame, line1, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, line2, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)

            if writer:
                writer.write(annotated_frame)

            if display:
                cv2.imshow("Sistema de Deteccion, Tracking y OCR (Grupo 33)", annotated_frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    print("Procesamiento detenido por el usuario.")
                    break

            if frame_idx % 200 == 0:
                print(
                    f"Fotograma {frame_idx}/{total_frames} | Personas: {len(unique_persons)} | "
                    f"Vehículos: {len(vehicle_classes)} | Matrículas: {total_plates_detected} | "
                    f"Leídas con OCR: {len(vehicle_plate_texts)}"
                )

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        df = pd.DataFrame(csv_records)
        if output_csv_path:
            column_names = [
                "fotograma", "tipo_objeto", "confianza", "identificador_tracking",
                "x1", "y1", "x2", "y2", "matrícula_en_su_caso", "confianza",
                "mx1", "my1", "mx2", "my2", "texto_matricula"
            ]
            df.to_csv(output_csv_path, index=False, header=column_names, encoding="utf-8")
            print(f"\nArchivo CSV guardado en: {output_csv_path}")

        print("\n=== RESUMEN FINAL DEL PROCESAMIENTO ===")
        print(f"Total fotogramas procesados: {frame_idx}")
        print(f"Total personas únicas detectadas: {len(unique_persons)}")
        print(f"Total vehículos únicos detectados: {len(vehicle_classes)}")
        for kind, count in vehicle_breakdown.items():
            print(f"  - {kind}: {count}")
        print(f"Flujo izquierda a derecha: {len(flow_left_to_right)}")
        print(f"Flujo derecha a izquierda: {len(flow_right_to_left)}")
        print(f"Vehículos que salieron por la derecha: {exits_right}")
        print(f"Vehículos que salieron por la izquierda: {exits_left}")
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
            "unique_persons": len(unique_persons),
            "unique_vehicles": len(vehicle_classes),
            "unique_vehicles_with_plate": len(unique_vehicles_with_plate),
            "recognized_plates_count": len(vehicle_plate_texts),
            "recognized_plates": dict(vehicle_plate_texts),
            "vehicle_breakdown": dict(vehicle_breakdown),
            "flow_l2r": len(flow_left_to_right),
            "flow_r2l": len(flow_right_to_left),
            "exits_right": exits_right,
            "exits_left": exits_left,
            "total_plates": len(unique_vehicles_with_plate)
        }
