# Registro de verificación · 21 de noviembre de 2026

## Entorno

Windows x64, Python 3.12.10. Entorno virtual aislado. `pip check`: **No broken requirements found**. Versiones directas en `requirements.txt`; versiones transitivas exactas en `requirements-lock.txt`.

## Modelo propio

Descarga JAFFE desde el registro oficial y checksum contrastado. Se prepararon 152 recortes de cinco clases: 92/30/30 y sujetos separados 6/2/2. Sin recortes rechazados ni duplicados exactos. Entrenamiento real HOG + StandardScaler + SVM lineal; C=0,01 seleccionado por validación. Accuracy test 0,4667 y F1 macro 0,4494. Resultados por clase y matriz de confusión en `reports/`.

Se comprobó que el scaler vio 92 muestras y que las predicciones tras guardar/cargar coinciden con las del pipeline para los 152 recortes. La extracción Haar compartida reproduce los mismos datos preparados. No se cambió el modelo después de mirar test. No se redistribuyen imágenes ni recortes JAFFE.

## Pruebas automatizadas

`python -m unittest discover -s tests -v`: **14 pruebas aprobadas**. Comprueban representación y artefacto, ausencia de cara, secuencias de una/dos manos, orden de manos, palmas cerradas, espera, pérdida de seguimiento, límite de partículas, WAV válidos, silencio, suavizado, recursos ausentes, reacción del filtro y límite de 30 s del grabador. `compileall` y ayuda de la aplicación correctos.

Los landmarks sintéticos comprueban reglas; no miden precisión del detector ni comodidad humana del gesto.

## Integración con imágenes y vídeo

Ambos modelos MediaPipe Tasks se cargaron en CPU. Hand Landmarker devolvió cero manos ante imagen vacía. Face Landmarker detectó 478 puntos en el retrato de LeBron de Erik Drost. La fotografía de acción de 2005 no produjo rostro con ese detector: una limitación con caras pequeñas/perfil.

`app.py` procesó un vídeo estático de 12 fotogramas en cada modo mediante `--headless --mute --record`. Se generaron `reports/smoke_mode1.avi` y `smoke_mode2.avi`, decodificables, y sus capturas finales. Se inspeccionaron panel, textos y corona, sin solapamiento con la cabecera. El modo 1 usó el modelo real y predijo happy en esta foto; esto no es una evaluación adicional de accuracy. Las imágenes `preview_mode*` son composiciones seleccionadas manualmente y lo indican en pantalla.

Los vídeos están marcados **PRUEBA TÉCNICA — FOTO ESTÁTICA — NO DEMO DE WEBCAM**. No sustituyen el vídeo de entrega. No contienen audio; los WAV se comprobaron estructuralmente sin evaluar su reproducción audible.

## Webcam y demostración pendientes

Se intentó abrir el índice 0 mediante OpenCV/DirectShow: `opened=false`, `frame_received=false`. No se capturó ni guardó imagen personal. No se han probado aquí expresiones reales, manos en movimiento, iluminación doméstica, latencia ni FPS de webcam. Es necesario ejecutar con la cámara del equipo de presentación, ajustar umbrales si hace falta y grabar ambas propuestas con R.

La implementación y los artefactos de entrenamiento están entregados. La demostración humana requiere una cámara disponible y no se presenta como completada.
