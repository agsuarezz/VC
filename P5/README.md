# LeBron Lab · práctica 5 · versión 5

Filtro local de LeBron para Windows con webcam, barba y corona. La aplicación incluye el clasificador de expresiones entrenado para la práctica. Este filtro reúne las reacciones faciales y los gestos de baloncesto en una vista. El segundo prototipo de la práctica queda por elegir.

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

La grabación se guarda en `recordings/`, como AVI MJPEG, con **máximo 30 segundos** y **sin audio**. Mantiene duración real mediante repetición de fotogramas si el ordenador va lento. Usa R o `--record recordings/demo.avi`. Para el vídeo de entrega cuando esté listo el segundo filtro, muestra ambos prototipos. Si quieres que se oigan los efectos de LeBron, usa una grabadora de pantalla que capture el audio del sistema.

## Entrenamiento mejorado

| Evaluación | V2 | V3 |
|---|---:|---:|
| WSEFEP, 40 fotos de ocho personas reservadas | 22/40 (55,0 %) | **33/40 (82,5 %)** |
| JAFFE histórico, 30 fotos de dos personas | 20/30 (66,7 %) | 20/30 (66,7 %) |

La SVM de la V3 aprende de JAFFE y WSEFEP, con **231 fotos reales de 30 personas** en desarrollo. Los rasgos combinan movimientos y geometría de MediaPipe. La selección por personas eligió usar copias reflejadas y con iluminación y nitidez modificadas durante el ajuste. El modelo final usa 693 ejemplos contando las copias. La webcam utiliza el modelo nuevo automáticamente. El resultado de WSEFEP se mide en fotos posadas y el rendimiento con webcam está por medir.

`models/emotion_v2.joblib` permite recuperar la V2 con `--model models/emotion_v2.joblib`. Las condiciones de uso, los comandos de reproducción y las métricas por clase están en [docs/DATASET.md](docs/DATASET.md).

## Comprobaciones y recursos

```powershell
python -m pip check
python -m unittest discover -s tests -v
python app.py --video TU_VIDEO.avi --headless --mute --record recordings/prueba.avi
```

La vista `reports/preview_lebron_v5.jpg` usa la fotografía de referencia de la astronauta Eileen Collins (NASA, dominio público). El programa dibuja la barba y la corona con OpenCV y reproduce sonidos sintetizados. [Registro de pruebas](docs/VALIDATION.md).

Modelos oficiales de Google y hashes: `models/sources.json`. Si faltan, ejecuta `python scripts/download_models.py`. MediaPipe Tasks funciona en CPU en modo IMAGE; la aplicación reutiliza una sola detección facial por fotograma para clasificar y componer. El rendimiento depende del equipo y no se ha medido FPS de webcam aquí.

El [enunciado](https://github.com/otsedom/otsedom.github.io/blob/main/VC/P5/README.md#tarea) pide dos escenarios que reaccionen al rostro, uno con entrenamiento propio y un vídeo máximo de 30 s. El filtro de LeBron usa el clasificador propio y reacciona a la cara y las manos. El segundo prototipo puede reaccionar a puntos faciales con reglas sencillas. Se revisaron los notebooks de HOG/SVM y MediaPipe; se conserva HOG como comparación reproducible y se amplía el clasificador con geometría. La precisión en webcam requiere una evaluación adicional con personas y sesiones nuevas.
