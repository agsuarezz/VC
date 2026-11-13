# Práctica 4 y 4b - Visión por Computador: Detección, Seguimiento y OCR de Vehículos y Matrículas

## Autoría

**Grupo 33**

- **Álvaro García Suárez** ([GitHub: @agsuarezz](https://github.com/agsuarezz))
- **Marcos González Gómez** ([GitHub: @mgz00](https://github.com/mgz00))

---

## Descripción del Proyecto

El objetivo de este proyecto integral (**Práctica 4 + Práctica 4b**) es desarrollar un sistema completo de visión artificial sobre secuencias de vídeo para la **detección y seguimiento persistente de vehículos y personas**, la **localización de matrículas**, el **reconocimiento óptico de caracteres (OCR) de las placas visibles**, el **conteo acumulado por categorías**, el **análisis del flujo direccional del tráfico**, la **anonimización de privacidad** y la **exportación estandarizada de métricas (vídeo anotado y CSV oficial de 15 columnas)**. Asímismo, se incluye el estudio experimental comparativo entre un **OCR clásico neuronal (`EasyOCR`)** y un **Modelo de Lenguaje Visual (`SmolVLM-Instruct`)**.

El trabajo aborda el **100% de los requisitos obligatorios** de ambas entregas (P4 y P4b), así como la totalidad de las tareas extras propuestas (**+1.75 puntos adicionales**):

- **Requisitos Obligatorios (P4 + P4b)**:
  - Detección y seguimiento persistente de personas y vehículos (`car`, `bus`, `truck`, `motorcycle`, `bicycle`) mediante **YOLO11** y **BoT-SORT** (`persist=True`).
  - Entrenamiento de un modelo detector de matrículas personalizado en formato YOLO a partir de un conjunto de datos real (`best_matriculas.pt`).
  - Localización de matrículas sobre vehículos en movimiento (filtrando vehículos estacionados con `VehicleMotionFilter`), asignación espacial al vehículo correspondiente y **reconocimiento OCR adaptativo con consenso multiframe** (`PlateOCR` en cascada `fast-plate-ocr` → `EasyOCR` → `SmolVLM-Instruct` con validación sintáctica española).
  - Conteo acumulado de instancias únicas y desglose por clases sobre la secuencia completa (2.832 fotogramas, 1 min 53 s).
  - Renderizado y volcado a disco del vídeo anotado completo con HUD en tiempo real, recuadros de seguimiento y **visualización de la matrícula leída** (`results/resultado_C0142_tracking.mp4`).
  - Generación del archivo CSV estructurado con el esquema exacto de 15 columnas incluyendo la columna `texto_matricula` completada (`results/resultado_C0142_detecciones.csv`).
  - **Comparativa empírica entre OCR Clásico (`EasyOCR`) y VLM (`SmolVLM-Instruct`)** sobre $N = 161$ matrículas verificadas de la base de datos de detección (completada con el texto Ground Truth donde faltaba), evaluando tasa de acierto y tiempos de inferencia.
- **Tareas Extras (+1.75 puntos)**:
  - **Extra 1 (+0.25p)**: Conteo del **flujo de coches hacia la cámara y hacia el fondo** en `C0142`.
  - **Extra 2 (+1.00p)**: Evaluación comparativa empírica entre **YOLO** y **RT-DETR** (arquitectura Transformer de Baidu con licencia Apache 2.0).
  - **Extra 3 (+0.25p)**: Detección de matrículas mediante **técnicas clásicas basadas en morfología matemática y contornos** en OpenCV.
  - **Extra 4 (+0.25p)**: Módulo conmutable de **anonimización de privacidad** (desenfoque gaussiano de personas y matrículas).

---

## Descarga y preparación

Para ejecutar el proyecto completo, descarga las carpetas **`P4`** y **`P4b`** y colócalas al mismo nivel, dentro de una misma carpeta. `P4b` utiliza archivos de `../P4`, así que conserva los nombres y la estructura de ambas carpetas.

En GitHub puedes descargar el repositorio desde **Code → Download ZIP** y descomprimirlo para conservar esa estructura. Para ejecutar el análisis de tráfico, coloca también el vídeo `C0142.MP4` dentro de `P4`.

## 1. Dataset de Matrículas y Preparación

Para entrenar el detector de matrículas se ha dispuesto de un conjunto de datos real de vehículos con matrículas españolas:
- **2.386 imágenes válidas** y sus correspondientes archivos JSON en `annotations/`.
- Más de **5.200 matrículas etiquetadas** con coordenadas normalizadas `[x_center, y_center, width, height]` y valor textual (`plate`).

A través del módulo `src/prepare_dataset.py`, las anotaciones se convirtieron al formato estándar de YOLO (`.txt` con clase `0`), dividiendo de forma estratificada y reproducible (`seed=42`) el dataset:
- **Entrenamiento (Train, 80%)**: 1.908 imágenes.
- **Validación (Val, 10%)**: 238 imágenes.
- **Prueba (Test, 10%)**: 240 imágenes.

```yaml
# dataset_matriculas.yaml
path: ./dataset
train: train/images
val: val/images
test: test/images

nc: 1
names: ['matricula']
```

---

## 2. Entrenamiento y Evaluación del Detector de Matrículas (YOLO11)

Tomando como punto de partida el modelo preentrenado `yolo11n.pt` de Ultralytics, se ejecutó un entrenamiento de fine-tuning sobre nuestra GPU **NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM)** durante 20 épocas con tamaño de imagen de 640 y batch de 16.

### Métricas de Rendimiento Obtenidas:

| Métrica | Valor Obtenido | Descripción |
| :--- | :---: | :--- |
| **Precisión (Precision, P)** | **84.9%** (0.849) | Proporción de detecciones predichas que son matrículas reales |
| **Exhaustividad (Recall, R)** | **58.3%** (0.583) | Capacidad para localizar las matrículas presentes en la escena |
| **mAP@50** | **60.0%** (0.600) | Precisión media con umbral de solapamiento IoU de 0.50 |
| **mAP@50-95** | **33.2%** (0.332) | Precisión media estricta ponderada en el rango IoU 0.50–0.95 |
| **Latencia de Inferencia** | **1.3 ms** (~770 FPS) | Tiempo de inferencia por fotograma en GPU RTX 4070 SUPER |

### Gráficas de Aprendizaje:

![Evolución del Entrenamiento](images/metricas_entrenamiento.png)
*Evolución de las funciones de pérdida (box_loss, cls_loss, dfl_loss) y de las métricas mAP durante las 20 épocas de entrenamiento.*

![Matriz de Confusión y Curva F1](images/confusion_matrix_matriculas.png)
*Matriz de confusión normalizada que evidencia la capacidad de discriminación del detector frente al fondo.*

Los pesos optimizados resultantes se almacenan en `best_matriculas.pt` y se integran en el pipeline de vídeo.

---

## 3. Comparativa: YOLO vs RT-DETR (Extra 2: +1.00p)

Como se describe en el apartado 4.5 del temario, recientes versiones de Ultralytics han adoptado la licencia **AGPL-3.0**, requiriendo licenciamiento de pago para uso comercial privativo. Como contrapunto, la industria y la literatura reciente han adoptado modelos como **RT-DETR (Baidu)** bajo licencia permisiva **Apache 2.0**, basados en arquitectura Transformer.

Se ha realizado una comparativa empírica ejecutando ambos modelos sobre la misma GPU RTX 4070 SUPER con resolución de 640x640:

| Característica | YOLO11 Nano (`yolo11n.pt`) | RT-DETR v2-S (`rtdetrv2_r18vd`) |
| :--- | :---: | :---: |
| **Desarrollador / Origen** | Ultralytics (2024) | Baidu Inc. (2023–2024) |
| **Arquitectura** | CNN con bloques residuales C3k2 + C2PSA | Vision Transformer (Backbone ResNet18-vd) |
| **Licencia de Software** | AGPL-3.0 (Copyleft fuerte) | **Apache 2.0 (Comercial abierta)** |
| **Latencia Inferencia (GPU)** | **5.80 ms** | **18.95 ms** |
| **Rendimiento (FPS)** | **~172 FPS** | **~53 FPS** |
| **Parámetros** | ~2.6 M | ~20 M |
| **Calidad de Detección** | Excelente en objetos principales | Excelente solidez en oclusiones y Transformers |

---

## 4. Detección Clásica de Matrículas por Contornos (Extra 3: +0.25p)

Se implementó un módulo complementario en OpenCV que prescinde de redes neuronales, explotando las propiedades geométricas y radiométricas de las placas de matrícula:

```mermaid
flowchart LR
    VehCrop["Recorte del Vehículo"] --> LowROI["Mitad Inferior (ROI)"]
    LowROI --> Bilat["Filtro Bilateral"]
    Bilat --> BlackHat["Morfología Black-Hat"]
    BlackHat --> SobelX["Gradiente Sobel X"]
    SobelX --> Otsu["Binarización Otsu & Clausura"]
    Otsu --> Contours["findContours & Filtrado Geométrico"]
    Contours --> PlateBox["Caja de Matrícula"]
```

1. **Restricción anatómica**: se aísla la mitad inferior del vehículo detectado.
2. **Morfología Black-Hat**: destaca zonas oscuras estrechas (números y letras) sobre fondo claro.
3. **Gradiente Sobel X**: acentúa los bordes verticales contrastados de la tipografía de la placa.
4. **Binarización de Otsu y Clausura**: agrupa los trazos de los caracteres en una masa rectangular unificada.
5. **Filtrado geométrico**: se verifica que la relación de aspecto $AR = \frac{\text{ancho}}{\text{alto}}$ esté dentro del rango típico de placas europeas ($2.2 \le AR \le 6.0$) y ocupe entre el 1% y el 35% del frontal/trasera del coche.

---

## 5. Detección, Seguimiento Persistente y OCR en Vídeo (`C0142.MP4`)

El sistema procesa la secuencia completa de evaluación `C0142.MP4` (**2.832 fotogramas, 113.28 segundos**) integrando detección, tracking y lectura OCR de matrículas en tiempo real:

1. **Seguimiento con BoT-SORT, reidentificación y filtro de vehículos estacionados (`VehicleMotionFilter`)**: las detecciones débiles pueden continuar un seguimiento y las de mayor confianza pueden iniciar uno nuevo. En la cámara fija de `C0142.MP4`, el filtro de movimiento (`VehicleMotionFilter`, que combina sustracción de fondo MOG2 con desplazamiento neto en ventana *lookahead* de 30 fotogramas y umbral mínimo de `30 px`) junto con la calibración de aparcamientos (`src/parking_identity.py`) descarta los vehículos aparcados para evitar falsos conteos y lecturas innecesarias.
2. **Asociación Vehículo–Matrícula y Reconocimiento OCR Adaptativo (`PlateOCR`)**: se asigna la placa al vehículo en movimiento que mejor contiene su recuadro. A medida que un vehículo se aproxima a cámara y su matrícula supera el umbral de legibilidad sin recorte en los bordes, `PlateOCR` ejecuta una cascada híbrida (`fast-plate-ocr` `cct-s-v2-global-model` → `EasyOCR` con ordenación por filas → `SmolVLM-Instruct` en recortes ambiguos) con acumulación de consenso multiframe por `track_id` y post-procesado sintáctico de matrículas españolas (`NNNNLLL` y provinciales `LLNNNNLL`), actualizando la etiqueta visual en pantalla (`Matricula #ID: TEXTO`) y la columna `texto_matricula` del CSV.
3. **HUD Informativo en Vivo**: monitorización gráfica superior que refleja fotogramas, personas, desglose de vehículos en movimiento, matrículas detectadas y leídas con OCR, y estadísticas de flujo direccional.

![Resultado de Seguimiento, OCR y HUD](images/resultado_tracking_ocr_hud.jpg)
*Captura de una ejecución anterior al cambio del contador de flujo. Muestra el seguimiento y la lectura de la matrícula en el fotograma 2141.*

### Resultados Globales sobre la Secuencia Completa:
- **Modelo General de Detección**: **YOLO11 Medium (`yolo11m.pt`)** con inferencia a alta resolución (**`imgsz=1280`**).
- **Fotogramas procesados**: 2.832 fotogramas (100% del vídeo).
- **Total personas únicas detectadas**: 91 IDs.
- **Total vehículos únicos detectados (en movimiento)**: **57 IDs** (`car`: 48, `motorcycle`: 4, `bicycle`: 3, `bus`: 2), tras filtrar **50 trayectorias estacionarias/aparcadas**.
- **Matrículas localizadas**: **38** (una por ID de vehículo en movimiento).
- **Matrículas visibles con texto reconocido por OCR**: **30 matrículas leídas** (`29` textos únicos; las 8 restantes corresponden a motocicletas o placas muy lejanas).
- **Flujo de coches**, calculado sobre las trayectorias guardadas en `results/resultado_C0142_detecciones.csv`:
  - Hacia la cámara por los carriles de la izquierda: **16**.
  - Hacia el fondo por los carriles de la derecha: **11**.
  - Sentido con más tráfico: **hacia la cámara**.

---

## 6. Extras: Flujo Direccional y Anonimización

### Extra 1 (+0.25p): Flujo de coches por sentido

- Se sigue el centro inferior de cada caja de coche. Cada `track_id` suma una vez al cruzar la línea transversal dibujada sobre la calzada: hacia abajo por el tramo izquierdo (hacia la cámara) o hacia arriba por el tramo derecho (hacia el fondo).
- El HUD y el resumen final muestran ambos contadores y el sentido con más tráfico. La posición de la línea está calibrada para la cámara fija de `C0142`.

### Extra 4 (+0.25p): Anonimización de Privacidad
- Modo conmutable de privacidad (`--anonymize`) mediante desenfoque gaussiano adaptativo (`cv2.GaussianBlur`) sobre la totalidad de personas y placas de matrícula localizadas.

![Fotograma Anonimizado](images/resultado_anonimizacion.jpg)
*Visualización del modo de privacidad activo: anonimización mediante desenfoque gaussiano de personas y matrículas.*

---

## 7. Archivo CSV Oficial de Resultados (con `texto_matricula`)

El procesado genera el fichero `results/resultado_C0142_detecciones.csv` con las **15 columnas exactas** estipuladas en las directrices de entrega:

```csv
fotograma,tipo_objeto,confianza,identificador_tracking,x1,y1,x2,y2,matrícula_en_su_caso,confianza,mx1,my1,mx2,my2,texto_matricula
```

- **Total de registros generados**: **18.794 detecciones** a lo largo de los 2.832 fotogramas (solo personas y vehículos en movimiento validados por `VehicleMotionFilter`).
- **Para personas**: los campos de matrícula quedan vacíos / `None`.
- **Para vehículos**: se registra con `Si` la detección de matrícula de cada `identificador_tracking`, sus coordenadas `mx1, my1, mx2, my2`, su confianza y el texto reconocido en `texto_matricula` (30 placas leídas de las 38 registradas).

```csv
fotograma,tipo_objeto,confianza,identificador_tracking,x1,y1,x2,y2,matrícula_en_su_caso,confianza,mx1,my1,mx2,my2,texto_matricula
1,bus,0.958,1,1395,52,1918,483,Si,0.568,1902,410,1919,427,3324HZC
401,car,0.935,41,1624,333,1919,533,Si,0.251,1903,412,1919,425,1502LSZ
535,car,0.765,42,1121,312,1182,345,Si,0.283,1162,333,1172,336,3713LKC
638,car,0.939,59,1622,336,1919,521,Si,0.695,1894,446,1919,464,5744GGP
875,car,0.83,73,1065,311,1139,344,Si,0.427,1123,333,1132,336,0848MCG
1125,car,0.778,77,1181,318,1216,343,Si,0.292,1194,333,1203,336,6713HCR
1271,car,0.759,95,1150,311,1201,357,Si,0.318,1163,341,1178,346,2914FGN
1395,car,0.759,120,1175,305,1247,367,Si,0.369,1204,341,1226,350,0053JFM
1486,car,0.763,141,1203,307,1271,366,Si,0.27,1228,347,1249,354,0285FKZ
1526,car,0.91,133,1011,309,1148,360,Si,0.304,1128,341,1141,345,3902FXD
1677,car,0.751,155,1155,307,1206,352,Si,0.284,1174,338,1188,342,8536LKJ
1936,truck,0.715,221,161,187,961,694,Si,0.266,250,565,415,619,5835LTT
2050,car,0.587,215,1108,302,1196,375,Si,0.539,1135,349,1161,355,1770JYG
2134,car,0.841,252,1123,312,1183,345,Si,0.264,1163,338,1176,342,8098JXM
2392,car,0.826,268,1120,313,1181,345,Si,0.254,1158,338,1172,343,1965KBP
2455,car,0.94,271,1538,336,1919,543,Si,0.633,1889,427,1920,445,GC5273CM
2681,car,0.917,294,1632,321,1919,510,Si,0.45,1904,407,1919,427,5017HDD
```

---

## 8. Práctica 4b: Comparativa Experimental entre OCR Clásico (`EasyOCR`) y VLM (`SmolVLM-Instruct`)

### 8.1 Especificaciones de la Máquina y Condiciones del Experimento

Todas las pruebas comparativas de OCR se han ejecutado bajo condiciones idénticas en la siguiente estación de trabajo:
- **CPU**: Intel® Core™ i7-14700KF (20 núcleos / 28 hilos)
- **GPU**: **NVIDIA GeForce RTX 4070 SUPER** (12.0 GB VRAM GDDR6X, CUDA 12.6)
- **RAM**: 32.0 GB DDR5 | **SO**: Windows 11 Pro 64-bit
- **Software**: Python 3.13, PyTorch `2.9.1+cu126`, `easyocr 1.7.2`, `transformers 5.22.2` (`bfloat16`, `sdpa`).
- **Dataset de Evaluación ($N = 161$ matrículas verificadas, `../P4b/ground_truth_matriculas.csv`)**: construido a partir de la base de datos de detección de matrículas de `dataset/`, combinando las 102 imágenes con matrícula en el nombre de archivo (verificadas visualmente) y completando manualmente el texto Ground Truth en 59 imágenes adicionales del split de `test` que carecían de etiqueta textual. Todos los recortes se pre-cargan en RAM y se mide la latencia pura de inferencia tras *warm-up* de GPU con `torch.cuda.synchronize()`.

### 8.2 Resultados Cuantitativos ($N = 161$ matrículas)

| Modelo / Configuración | Acierto Exacto Placa (%) | Acierto por Carácter ($1-\text{CER}$) | Similitud Media (%) | Distancia Levenshtein Media | Latencia Media (ms) | Latencia Mediana (ms) | Latencia P95 (ms) | Throughput (placas/s) | VRAM Pico (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EasyOCR (Crudo)** | 49.07% | 64.10% | 82.82% | 2.522 | **27.19 ms** | **22.06 ms** | **63.38 ms** | **36.77 img/s** | **378.8 MB** |
| **EasyOCR (+Post-proc. ES)** | 59.01% | 77.90% | 84.54% | 1.553 | **27.19 ms** | **22.06 ms** | **63.38 ms** | **36.77 img/s** | **378.8 MB** |
| **SmolVLM-Instruct (Crudo)** | **89.44%** | 95.58% | 96.39% | 0.311 | 409.11 ms | 409.69 ms | 561.71 ms | 2.44 img/s | 4.882,5 MB |
| **SmolVLM-Instruct (+Post-proc. ES)** | **89.44%** | **95.67%** | **96.48%** | **0.304** | 409.11 ms | 409.69 ms | 561.71 ms | 2.44 img/s | 4.882,5 MB |

![Comparativa de Rendimiento y Tiempos OCR vs VLM](images/comparativa_ocr_rendimiento_tiempos.png)
*Comparativa de tasa de acierto exacto y por carácter (izquierda) y tiempos de inferencia en GPU RTX 4070 SUPER (derecha).*

![Análisis por Resolución y Errores de Edición](images/comparativa_ocr_resolucion_errores.png)
*Desglose de tasa de acierto según resolución en píxeles de la matrícula (izquierda) y distribución de errores de edición Levenshtein (derecha).*

![Ejemplos Visuales del Benchmark OCR vs VLM](images/ejemplos_comparativa_ocr.png)
*Ejemplos visuales del benchmark mostrando casos de acierto conjunto y casos con inclinación (`3053GLD`), ruido de eurobanda (`5334LPR`) o desgaste tipográfico (`3050KGM`, `4691JTZ`) donde SmolVLM supera a EasyOCR.*

### 8.3 Conclusiones de la Comparativa OCR vs VLM:
1. **Velocidad y Ligereza vs. Exactitud**: **EasyOCR** es **$\sim 15\text{ veces más rápido}$** (**27.19 ms** vs **409.11 ms**) y requiere **$\sim 13\text{ veces menos VRAM}$** (**378.8 MB** vs **4.88 GB**), mientras que **SmolVLM-Instruct** logra una ventaja de **+30.43 puntos porcentuales en acierto exacto de matrícula completa** (**89.44%** frente a **59.01%**) y alcanza un **95.67% de precisión por carácter**.
2. **Robustez ante Perspectiva y Baja Resolución**: en placas inclinadas (`3053GLD`), el detector de líneas de EasyOCR divide y desordena los bloques (`GLD3053`), y en placas pequeñas ($<220\text{ px}$) su acierto cae al **35.71%**, mientras que SmolVLM mantiene un **78.57%** en placas pequeñas y **>91%** en medianas y grandes.
3. **Sinergia en Vídeo**: el uso combinado de seguimiento persistente (**BoT-SORT**), filtrado de vehículos estacionados (`VehicleMotionFilter`) e inferencia OCR en cascada (`fast-plate-ocr` → `EasyOCR` → `SmolVLM-Instruct` con consenso multiframe por `track_id`) permite alcanzar la máxima fiabilidad (`30/38` matrículas en movimiento leídas en `C0142.MP4`) sin comprometer la velocidad global del sistema.

---

## 9. Estructura de Ficheros y Entregables

```
VC/
├── P4/
│   ├── VC_P4_Entrega_Grupo33.ipynb            # Cuaderno Jupyter principal (Detección, Tracking y OCR)
│   ├── README.md                              # Memoria técnica integrada (P4 + P4b)
│   ├── best_matriculas.pt                     # Pesos entrenados del detector de matrículas YOLO11
│   ├── botsort_trafico.yaml                   # Configuración del tracker BoT-SORT
│   ├── src/
│   │   ├── traffic_pipeline.py                # Núcleo modular (tracking, filtro movimiento, contornos, OCR híbrido, HUD, flujo)
│   │   ├── parking_identity.py                # Calibración espacial de plazas estacionadas en escena fija
│   │   ├── run_pipeline.py                    # CLI para procesar secuencias con soporte --ocr y filtro de movimiento
│   │   ├── prepare_dataset.py                 # Preparación y partición Train/Val/Test del dataset YOLO
│   │   ├── train_plates.py                    # Script de entrenamiento YOLO11 de matrículas
│   │   └── load_rtdetr.py                     # Cargador de RT-DETR v1/v2 para comparativa
│   ├── results/
│   │   ├── resultado_C0142_tracking.mp4       # Vídeo anotado completo con matrículas leídas en pantalla
│   │   ├── resultado_C0142_detecciones.csv    # CSV oficial con las 15 columnas y texto_matricula completado
│   │   ├── resultado_C0142_anonimizado.mp4    # Vídeo completo en modo anonimización de privacidad
│   │   └── resultado_C0142_anonimizado.csv    # CSV generado en modo anonimizado
│   └── images/                                # Figuras y gráficas de evaluación
└── P4b/
    ├── README.md                              # Memoria específica y comparativa detallada de Práctica 4b
    ├── VC_P4b.ipynb                           # Cuaderno de demostración de OCRs, VLM y benchmark comparativo
    ├── ground_truth_matriculas.csv            # 161 matrículas verificadas con texto Ground Truth completado
    ├── src/
    │   └── benchmark_ocr.py                   # Script reproducible del experimento EasyOCR vs SmolVLM
    ├── results/
    │   ├── resultados_comparativa_ocr.csv     # Predicciones, métricas y latencias detalladas por imagen
    │   ├── resumen_metricas_ocr.csv           # Tabla resumen de métricas del experimento
    │   └── resultado_C0142_detecciones.csv    # Copia del CSV oficial con texto_matricula completado
    └── images/                                # Gráficas comparativas y capturas visuales de P4b
```

---
Bajo licencia de Creative Commons Reconocimiento - No Comercial 4.0 Internacional.
