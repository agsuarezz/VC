# Verificación de la versión 2 · 21 de noviembre de 2026

## Entrenamiento

Modelo propio entrenado y guardado. La v2 combina 52 coeficientes de movimiento facial y 76 coordenadas normalizadas con SVM lineal. Selección por GroupKFold4 con ocho personas de desarrollo, sin compartir personas entre entrenamiento y validación en cada fold. El modelo seleccionado se reajusta con122 muestras de desarrollo; compara una sola vez la elección final con30 imágenes de dos personas reservadas del benchmark histórico.

Accuracy histórica20/30=66,67% frente14/30=46,67%; F1 macro0,6667 frente0,4494. Es un conjunto pequeño previamente consultado, por lo que **no se declara un test nuevo intacto**. No se eligieron otros cambios tras consultar el resultado final. No demuestra precisión de webcam. La baseline y su modelo se conservan por separado. Métricas y discusión completas en DATASET.md y reports/metrics.json.

La inferencia reutiliza la misma detección de MediaPipe del compositor. Los coeficientes se ordenan por nombre y la geometría se normaliza respecto al eje de ojos y su distancia, corrigiendo la relación de aspecto del fotograma.

## Composición

Fondo a lienzo completo, sin panel lateral ni webcam de fondo. Alineación de ojos, escala uniforme, rotación, máscara convexa y suavizado; corrección parcial de color. Los landmarks de destino son reales y precalculados. Vista técnica inspeccionada con fotografía de NASA sobre LeBron en reports/preview_composite.jpg, marcada como prueba, sin capturas personales.

Aplicación ejecutada en modo1 con modelo v2 y vídeo estático de12 fotogramas: termina correctamente, genera reports/composite_v2_smoke.avi decodificable. Modo2 ejecutado con el mismo vídeo: correcto, reports/gestures_v2_smoke.avi. Un vídeo estático no prueba expresiones o gestos humanos.

## Gestos

13 pruebas específicas del módulo de gestos aprobadas. Incluyen dos palmas elevadas mantenidas0,35s sin preparación baja, pérdida breve y larga de detección, prioridad sobre tiro, bloqueo de repeticiones, rearmado visible y baja tasa de fotogramas. La nube tiene más partículas y mayor duración. La aplicación dibuja landmarks de manos y ofrece mensajes de ayuda y progreso. La corrección de reglas no se presenta como validación con una persona.

## Entorno y límites

Windows x64, Python3.12.10, mismas dependencias verificadas de v1. Pruebas completas y número final en el registro de ejecución adjunto. No se han añadido paquetes. Los sonidos se comprobaron como WAV válidos y con silencio, no se evalúa aquí la reproducción audible. La ejecución gráfica de pantalla completa requiere probarla con la pantalla y webcam del usuario; se puede volver a ventana mediante F o --windowed.

El primer registro de14 pruebas y el intento de cámara no disponible se conserva en VALIDATION_V1.md. Para completar la comprobación con cámara: ejecutar iniciar.bat, verificar que el modo1 coloca la cara, mantener dos palmas junto a la cara en modo2 hasta100%, bajar/cerrar ambas y repetir. La grabación R permite preparar la demostración final de máximo30 segundos.
