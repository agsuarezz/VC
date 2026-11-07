import os
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO

COCO_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
COCO_PERSON_CLASS = 0

def detect_plate_contours(vehicle_bgr):
    """
    Detector clásico de matrículas mediante morfología matemática y contornos (Extra 3).
    Busca regiones rectangulares con contraste típico de placa en el tercio inferior del vehículo.
    """
    vh, vw = vehicle_bgr.shape[:2]
    if vh < 30 or vw < 40:
        return None
        
    # Analizamos la mitad inferior del vehículo donde se ubican las placas
    y_start = int(vh * 0.45)
    roi = vehicle_bgr[y_start:, :]
    rh, rw = roi.shape[:2]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Kernel rectangular alargado para destacar caracteres de matrícula
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
    
    # Gradiente horizontal (Sobel X) para resaltar bordes verticales de los caracteres
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    minVal, maxVal = np.min(gradX), np.max(gradX)
    if maxVal > minVal:
        gradX = ((gradX - minVal) / (maxVal - minVal) * 255).astype("uint8")
    else:
        gradX = gradX.astype("uint8")
        
    # Binarización Otsu y clausura para agrupar caracteres en una única masa rectangular
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
        # Rango de aspecto típico de matrículas europeas: ~2.5 a ~5.8
        if 2.2 <= ar <= 6.0:
            area = w * h
            roi_area = rw * rh
            rel_area = area / roi_area
            # Debe ocupar una porción razonable del frontal/trasera del coche
            if 0.01 <= rel_area <= 0.35:
                # Puntuamos mayor área y aspecto cercano a 4.5
                score = area * (1.0 - abs(ar - 4.5) / 4.5)
                if score > best_score:
                    best_score = score
                    # Convertir coordenadas relativas a la ROI en coordenadas del vehículo
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
            # Kernel proporcional al tamaño del objeto
            kw = (bw // 2) * 2 + 1
            kh = (bh // 2) * 2 + 1
            kw = max(15, min(99, kw))
            kh = max(15, min(99, kh))
            roi = out[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (kw, kh), 30)
            out[y1:y2, x1:x2] = blurred_roi
    return out


class TrafficAnalyzer:
    """
    Sistema integral de detección, tracking, análisis de flujo y localización de matrículas.
    """
    def __init__(self, general_model_path="yolo11m.pt", plate_model_path=None, conf_thres=0.35, imgsz=1280, device="cuda:0"):
        print(f"Inicializando TrafficAnalyzer con {general_model_path} (imgsz={imgsz}) en {device}...")
        self.device = device
        self.conf_thres = conf_thres
        self.imgsz = imgsz
        
        # Detector general de personas y vehículos (YOLO11 Medium para máxima precisión)
        self.general_model = YOLO(general_model_path)
        
        # Detector entrenado de matrículas
        self.plate_model = YOLO(plate_model_path) if plate_model_path and os.path.exists(plate_model_path) else None
        if self.plate_model:
            print(f"Detector de matrículas cargado: {plate_model_path}")
        else:
            print("Detector de matrículas por Deep Learning no disponible aún. Usando método de contornos.")

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
        unique_vehicles = defaultdict(set) # class_name -> set(track_ids)
        all_unique_vehicles = set()
        unique_vehicles_with_plate = set()
        
        # Estadísticas de flujo direccional (Extra 1)
        flow_left_to_right = set()
        flow_right_to_left = set()
        exits_left = 0
        exits_right = 0
        exited_ids = set()
        
        # Conteo de matrículas detectadas
        total_plates_detected = 0
        
        # Registros para el archivo CSV oficial
        csv_records = []
        
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
                tracker="botsort.yaml",
                device=self.device,
                verbose=False
            )
            
            current_vehicles = [] # [(vx1, vy1, vx2, vy2, track_id, conf, class_name), ...]
            current_persons = []
            
            if track_results[0].boxes is not None and track_results[0].boxes.id is not None:
                boxes = track_results[0].boxes.xyxy.cpu().numpy()
                track_ids = track_results[0].boxes.id.int().cpu().tolist()
                confidences = track_results[0].boxes.conf.cpu().numpy()
                clses = track_results[0].boxes.cls.int().cpu().tolist()
                
                for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, clses):
                    x1, y1, x2, y2 = box.astype(int)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Actualizar historial de trayectoria
                    track_history[track_id].append((cx, y2))
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)
                        
                    # Clasificar objeto
                    if cls_id == COCO_PERSON_CLASS:
                        unique_persons.add(track_id)
                        current_persons.append((x1, y1, x2, y2, track_id, conf, "person"))
                        if anonymize:
                            boxes_to_blur.append((x1, y1, x2, y2))
                    elif cls_id in COCO_VEHICLE_CLASSES:
                        cname = COCO_VEHICLE_CLASSES[cls_id]
                        unique_vehicles[cname].add(track_id)
                        all_unique_vehicles.add(track_id)
                        current_vehicles.append((x1, y1, x2, y2, track_id, conf, cname))
                        
                    # 2. Análisis de Flujo Direccional (Extra 1)
                    trail = track_history[track_id]
                    if len(trail) >= 8:
                        dx = trail[-1][0] - trail[0][0]
                        if dx > 35:
                            flow_left_to_right.add(track_id)
                        elif dx < -35:
                            flow_right_to_left.add(track_id)
                            
                    # Comprobación de salidas de la escena por los laterales
                    if track_id not in exited_ids and len(trail) >= 5:
                        if cx > width - 60 and trail[-1][0] > trail[-3][0]:
                            exits_right += 1
                            exited_ids.add(track_id)
                        elif cx < 60 and trail[-1][0] < trail[-3][0]:
                            exits_left += 1
                            exited_ids.add(track_id)
            
            # 3. Detección y Asociación de Matrículas (máximo 1 sola matrícula por vehículo)
            plates_in_frame = {} # vehicle_track_id -> (mx1, my1, mx2, my2, conf_m, texto_m)
            
            # Opción A: Detección con modelo YOLO entrenado
            if self.plate_model:
                plate_results = self.plate_model(frame, conf=0.25, imgsz=self.imgsz, device=self.device, verbose=False)
                if plate_results[0].boxes is not None and len(plate_results[0].boxes) > 0:
                    pboxes = plate_results[0].boxes.xyxy.cpu().numpy()
                    pconfs = plate_results[0].boxes.conf.cpu().numpy()
                    for pbox, pconf in zip(pboxes, pconfs):
                        px1, py1, px2, py2 = pbox.astype(int)
                        pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                        
                        # Asociar al vehículo que contenga el centro de la placa
                        best_veh_id = None
                        for (vx1, vy1, vx2, vy2, v_id, _, _) in current_vehicles:
                            if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                                best_veh_id = v_id
                                break
                        if best_veh_id:
                            # REGLA: Máximo 1 sola matrícula por vehículo.
                            # Si ya existe una candidata dentro del coche, conservamos la de mayor confianza.
                            if best_veh_id not in plates_in_frame or float(pconf) > plates_in_frame[best_veh_id][4]:
                                plates_in_frame[best_veh_id] = (px1, py1, px2, py2, float(pconf), "")
                                
            # Opción B: Si no se detectó con YOLO, probar técnica clásica por contornos (Extra 3)
            if use_contour_plates:
                for (vx1, vy1, vx2, vy2, v_id, _, _) in current_vehicles:
                    # Solo evaluamos si el vehículo aún no tiene ninguna matrícula asignada
                    if v_id not in plates_in_frame:
                        vcrop = frame[vy1:vy2, vx1:vx2]
                        cbox = detect_plate_contours(vcrop)
                        if cbox:
                            cx1, cy1, cx2, cy2 = cbox
                            abs_px1 = vx1 + cx1
                            abs_py1 = vy1 + cy1
                            abs_px2 = vx1 + cx2
                            abs_py2 = vy1 + cy2
                            plates_in_frame[v_id] = (abs_px1, abs_py1, abs_px2, abs_py2, 0.65, "Contorno")

            # Actualizar conteo total de detecciones y vehículos únicos con matrícula identificada
            total_plates_detected += len(plates_in_frame)
            unique_vehicles_with_plate.update(plates_in_frame.keys())
            if anonymize:
                for (mx1, my1, mx2, my2, _, _) in plates_in_frame.values():
                    boxes_to_blur.append((mx1, my1, mx2, my2))

            # 4. Registrar filas para el CSV Oficial
            # Para cada persona detectada
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
                
            # Para cada vehículo detectado
            for (vx1, vy1, vx2, vy2, v_id, v_conf, v_cls) in current_vehicles:
                if v_id in plates_in_frame:
                    mx1, my1, mx2, my2, mconf, mtext = plates_in_frame[v_id]
                    has_plate = "Si"
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

            # 5. Aplicar anonimización de privacidad si está activa (Extra 4)
            if anonymize and boxes_to_blur:
                annotated_frame = apply_anonymization(annotated_frame, boxes_to_blur)

            # 6. Renderizado de anotaciones visuales
            # A) Dibujar personas
            for (x1, y1, x2, y2, p_id, conf, _) in current_persons:
                color = (255, 128, 0) # Naranja
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                lbl = f"Persona #{p_id} ({conf:.2f})"
                cv2.putText(annotated_frame, lbl, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                # Estela
                pts = np.array(track_history[p_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], isClosed=False, color=color, thickness=2)

            # B) Dibujar vehículos y sus matrículas
            for (vx1, vy1, vx2, vy2, v_id, v_conf, v_cls) in current_vehicles:
                np.random.seed(v_id * 53 + 7)
                vcolor = tuple(int(c) for c in np.random.randint(50, 230, 3).tolist())
                cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), vcolor, 2)
                
                v_lbl = f"{v_cls} #{v_id} ({v_conf:.2f})"
                cv2.putText(annotated_frame, v_lbl, (vx1, max(20, vy1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, vcolor, 2, cv2.LINE_AA)
                
                # Estela
                pts = np.array(track_history[v_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], isClosed=False, color=vcolor, thickness=2)
                
                # Matrícula asociada
                if v_id in plates_in_frame:
                    mx1, my1, mx2, my2, mconf, _ = plates_in_frame[v_id]
                    cv2.rectangle(annotated_frame, (mx1, my1), (mx2, my2), (0, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Matricula", (mx1, max(15, my1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # 7. Renderizado del HUD Superior Profesional
            hud_bg = np.zeros((75, width, 3), dtype=np.uint8)
            cv2.rectangle(hud_bg, (0, 0), (width, 75), (20, 20, 20), -1)
            cv2.line(hud_bg, (0, 74), (width, 74), (0, 180, 255), 2)
            annotated_frame[0:75, 0:width] = cv2.addWeighted(annotated_frame[0:75, 0:width], 0.25, hud_bg, 0.75, 0)
            
            line1 = f"Fotograma: {frame_idx:04d}/{total_frames} | Personas Activas: {len(current_persons)} (Total: {len(unique_persons)}) | Vehiculos Activos: {len(current_vehicles)} (Total: {len(all_unique_vehicles)})"
            veh_desglose = " | ".join([f"{k.capitalize()}s: {len(v)}" for k, v in unique_vehicles.items() if len(v) > 0])
            line2 = f"Desglose: {veh_desglose if veh_desglose else 'None'} | Matriculas: {total_plates_detected} | Flujo Dir: [Der->Izq: {len(flow_right_to_left)}, Izq->Der: {len(flow_left_to_right)}]"
            
            cv2.putText(annotated_frame, line1, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, line2, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)

            if writer:
                writer.write(annotated_frame)
                
            if display:
                cv2.imshow("Sistema de Detección y Seguimiento (Grupo 33)", annotated_frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    print("Procesamiento detenido por el usuario.")
                    break
                    
            if frame_idx % 100 == 0:
                print(f"Fotograma {frame_idx}/{total_frames} | Personas: {len(unique_persons)} | Vehículos: {len(all_unique_vehicles)} | Matrículas: {total_plates_detected}")

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
            
        # Exportar DataFrame a CSV con el esquema exacto de la práctica
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
        print(f"Total vehículos únicos detectados: {len(all_unique_vehicles)}")
        for k, v in unique_vehicles.items():
            print(f"  - {k}: {len(v)}")
        print(f"Flujo izquierda a derecha: {len(flow_left_to_right)}")
        print(f"Flujo derecha a izquierda: {len(flow_right_to_left)}")
        print(f"Vehículos que salieron por la derecha: {exits_right}")
        print(f"Vehículos que salieron por la izquierda: {exits_left}")
        print(f"Total detecciones de matrícula: {total_plates_detected}")
        print(f"Vehículos únicos con matrícula identificada: {len(unique_vehicles_with_plate)}")
        if output_video_path:
            print(f"Vídeo de salida exportado en: {output_video_path}")
            
        return {
            "df": df,
            "unique_persons": len(unique_persons),
            "unique_vehicles": len(all_unique_vehicles),
            "unique_vehicles_with_plate": len(unique_vehicles_with_plate),
            "vehicle_breakdown": {k: len(v) for k, v in unique_vehicles.items()},
            "flow_l2r": len(flow_left_to_right),
            "flow_r2l": len(flow_right_to_left),
            "exits_right": exits_right,
            "exits_left": exits_left,
            "total_plates": total_plates_detected
        }
