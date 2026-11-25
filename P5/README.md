# LeBron Lab & Arte Generativo Facial · práctica 5 · versión 5

Aplicación local para Windows con webcam para la Práctica 5:

1. **Prototipo 1 — LeBron Lab (`app.py` / `iniciar.bat`):** barba y corona sobre el rostro, clasificador de expresiones entrenado para la práctica y gestos de baloncesto con audio.
2. **Prototipo 2 — Arte Generativo Facial (`app_arte.py` / `iniciar_arte.bat`):** lienzo visual reactivo inspirado en *Más Que la Cara* (Zach Lieberman) que transforma tus landmarks 3D y `blendshapes` en gráficos en tiempo real con 3 modos artísticos.

## Hechizos animales

Ejecuta `iniciar_mago.bat`. Deja solo el índice extendido y dibuja una figura grande en el aire. Vuelve al punto inicial y baja el dedo para preparar otro hechizo. También puedes bajar el dedo para terminar un trazo abierto y ver el efecto de fallo.

- Círculo → conejo lunar
- Cuadrado → oso del bosque
- Triángulo → zorro de fuego
- Trazo irreconocible → rasgos mezclados

Pulsa **C** para quitar la transformación, **R** para iniciar o detener una grabación de hasta 30 segundos y **Q** para salir. Puedes usar este filtro como segundo prototipo junto a LeBron.

## Abrir

En la carpeta `P5`, haz doble clic en **iniciar.bat**. Desde PowerShell, dentro de esa carpeta:

```powershell
.\.venv\Scripts\python.exe app.py
```

Abre a pantalla completa. `F` alterna con ventana; `H` oculta o muestra ayuda. Para elegir otra cámara, añade `--camera 1`. Cierra otras aplicaciones que estén usando la webcam.

Para una copia extraída del ZIP, abre PowerShell en la carpeta que contiene `app.py` e instala el entorno (Python **3.12 x64**, comprobado con 3.12.10):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Si el lanzador `py` no encuentra Python, usa la ruta completa de Python 3.12 en el primer comando. No se necesita activar el entorno. `requirements-lock.txt` conserva las versiones transitivas probadas en Windows. No instales también `opencv-python`: `opencv-contrib-python` ya proporciona `cv2`.

## Filtro de LeBron

Al detectar tu cara, aparecen una barba oscura inspirada en LeBron y una corona dorada. Los accesorios siguen la mandíbula, la frente y el giro de tu cabeza. La boca queda visible y el fondo sigue siendo tu cámara.

| Expresión | Sonido |
|---|---|
| Alegría | Tres campanillas ascendentes |
| Tristeza | Tres notas descendentes |
| Enfado | Golpe grave |
| Sorpresa | Tono ascendente rápido |
| Neutral | Silencio |

Cada sonido se activa una vez cuando la predicción se estabiliza. Cambia de expresión o relaja la cara para repetirlo. Los WAV se generan localmente y se pueden sustituir en `assets/audio/` por tus propios sonidos con los mismos nombres.

- **Tiro.** Sube una mano desde el pecho hasta cerca de la cara como si lanzaras. Aparece el balón y se oye el efecto.
- **Talco.** Empieza con ambas manos bajas. Súbelas juntas y abre ambas palmas al llegar a la altura de la cara. Se dibuja la nube de talco.

La ayuda muestra cuántas manos detecta y el estado del gesto. [Cómo funcionan los gestos](docs/GESTURES.md).

## Controles

| Tecla | Acción |
|---|---|
| `F` | Pantalla completa / ventana |
| `H` | Mostrar / ocultar ayuda |
| `M` | Silenciar / activar audio |
| `R` | Iniciar / detener grabación |
| `Q` / `Esc` | Salir |

`--windowed` inicia en ventana. Los WAV son efectos originales sintetizados; se reproducen con winsound en Windows. No requieren descargas de sonidos comerciales. En otros sistemas el audio puede quedar deshabilitado.

La grabación se guarda en `recordings/`, como AVI MJPEG, con **máximo 30 segundos** y **sin audio**. Mantiene duración real mediante repetición de fotogramas si el ordenador va lento. Usa R o `--record recordings/demo.avi`. Para el vídeo de entrega muestra ambos prototipos (`app.py` y `app_arte.py`). Si quieres que se oigan los efectos de LeBron, usa una grabadora de pantalla que capture el audio del sistema.

## Arte Generativo Facial (Prototipo 2)

Inspirado en *Más Que la Cara* (Zach Lieberman), este segundo prototipo convierte tu rostro en un lienzo visual interactivo usando los 478 landmarks 3D y los 52 `blendshapes` de `face_landmarker.task`.

Para abrirlo haz doble clic en **iniciar_arte.bat** o ejecuta:

```powershell
python app_arte.py
```

Reacciones faciales implementadas:

- **Abrir la boca:** emite ondas concéntricas (o caracteres en modo tipográfico) cuya escala y velocidad crecen con la apertura mandibular.
- **Levantar las cejas:** dispara destellos luminosos desde ojos y cejas.
- **Inclinar o mover la cabeza:** rota la paleta de colores HSV, activa estelas temporales (*motion trail*) y genera paralaje en el modo 3D.
- **Guiñar un ojo:** alterna entre los 3 modos artísticos (ignora parpadeos normales de ambos ojos y aplica *cooldown*):
  1. **Neón + Ondas:** contornos faciales con brillo *glow*, iris luminosos, estela y ondas desde la boca.
  2. **Tipografía Facial:** silueta dibujada con caracteres dinámicos que reaccionan al habla/apertura bucal.
  3. **Topografía 3D:** malla y nodos coloreados por profundidad `z` con efecto paralaje.

Controles específicos de `app_arte.py`: `1`/`2`/`3` (o guiño) cambian de modo, `B` alterna entre cámara atenuada y fondo negro puro, `H` oculta/muestra telemetría, `F` pantalla completa, `R` graba hasta 30 s en `recordings/` y `Q`/`Esc` sale.

```powershell
python app_arte.py --video TU_VIDEO.avi --headless --max-frames 60 --record recordings/prueba_arte.avi
```

## Entrenamiento con FER2013

El filtro abre por defecto el modelo de FER2013. Se ajustó con 20.388 fotos con rasgos faciales disponibles y logró 67,6 % de acierto frente al 44,5 % de V3 en las mismas 2.689 fotos de prueba con rasgos disponibles. La prueba mide clasificación en FER2013. El resultado en webcam queda por medir. [Método](docs/DATASET.md) · [Métricas](reports/metrics_fer2013.json).

Para abrir V3, usa `iniciar.bat --model models/emotion.joblib`.

## Entrenamiento previo con JAFFE y WSEFEP

| Evaluación | V2 | V3 |
|---|---:|---:|
| WSEFEP, 40 fotos de ocho personas reservadas | 22/40 (55,0 %) | **33/40 (82,5 %)** |
| JAFFE histórico, 30 fotos de dos personas | 20/30 (66,7 %) | 20/30 (66,7 %) |

La SVM de la V3 aprende de JAFFE y WSEFEP, con **231 fotos reales de 30 personas** en desarrollo. Los rasgos combinan movimientos y geometría de MediaPipe. La selección por personas eligió usar copias reflejadas y con iluminación y nitidez modificadas durante el ajuste. El modelo final usa 693 ejemplos contando las copias. El resultado de WSEFEP se mide en fotos posadas y el rendimiento con webcam está por medir.

`models/emotion_v2.joblib` permite recuperar la V2 con `--model models/emotion_v2.joblib`. Las condiciones de uso, los comandos de reproducción y las métricas por clase están en [docs/DATASET.md](docs/DATASET.md).

## Comprobaciones y recursos

```powershell
python -m pip check
python -m unittest discover -s tests -v
python app.py --video TU_VIDEO.avi --headless --mute --record recordings/prueba.avi
```

La vista `reports/preview_lebron_v5.jpg` usa la fotografía de referencia de la astronauta Eileen Collins (NASA, dominio público). El programa dibuja la barba y la corona con OpenCV y reproduce sonidos sintetizados. [Registro de pruebas](docs/VALIDATION.md).

Modelos oficiales de Google y hashes: `models/sources.json`. Si faltan, ejecuta `python scripts/download_models.py`. MediaPipe Tasks funciona en CPU en modo IMAGE; la aplicación reutiliza una sola detección facial por fotograma para clasificar y componer. El rendimiento depende del equipo y no se ha medido FPS de webcam aquí.

El [enunciado](https://github.com/otsedom/otsedom.github.io/blob/main/VC/P5/README.md#tarea) pide dos escenarios que reaccionen al rostro, uno con entrenamiento propio y un vídeo máximo de 30 s. El filtro de LeBron (`app.py`) usa el clasificador propio y reacciona a la cara y las manos; el segundo prototipo (`app_arte.py`) convierte los landmarks 3D y `blendshapes` faciales en arte generativo interactivo. Se revisaron los notebooks de HOG/SVM y MediaPipe; se conserva HOG como comparación reproducible y se amplía el clasificador con geometría. La precisión en webcam requiere una evaluación adicional con personas y sesiones nuevas.
