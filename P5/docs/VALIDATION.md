# Verificación de la versión 5

- 22 pruebas aprobadas en Windows y Python 3.12.10. La prueba visual usa puntos reales de MediaPipe sobre una foto de referencia y comprueba que la boca mantiene sus píxeles y que la barba y la corona afectan a las regiones esperadas.
- La aplicación procesó 75 fotogramas con el modelo de expresiones. La foto fija desencadenó una sola llamada al sonido de sorpresa. La salida conserva el encuadre original y dibuja los accesorios. La captura se guarda en `reports/preview_lebron_v5.jpg`.
- Se verificaron seis WAV diferentes, cuatro expresiones, tiro y talco. Se comprobó el disparo del sonido por código. La reproducción audible y la comodidad de los gestos quedan por probar con la webcam del usuario. Las reglas de tiro y talco se verificaron con secuencias de puntos.
- El modelo de expresiones conserva el entrenamiento de V3 y sus métricas en `docs/DATASET.md` y `reports/metrics.json`.
