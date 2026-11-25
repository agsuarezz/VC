# Dataset, entrenamiento y evaluación

## Reentrenamiento con FER2013

Se usó el CSV de FER2013 con las particiones originales `Training`, `PublicTest` y `PrivateTest`, reservadas para ajuste, validación y prueba, respectivamente. Se conservaron las cinco etiquetas del filtro y se retiraron 1.446 duplicados exactos dando prioridad a prueba y validación. FER2013 carece de identificadores de persona. Pueden quedar sujetos compartidos y fotos casi duplicadas. [Competición original de Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) · [Copia del CSV usada](https://www.kaggle.com/datasets/deadskull7/fer2013).

El modelo usa los mismos 128 rasgos de MediaPipe que V3. Se comparan regresión logística y árboles con boosting, variando la regularización, el número de hojas y las etapas. Entre los candidatos a menos de dos puntos de F1 macro del mejor en validación se elige el de menor brecha absoluta de F1 entre ajuste y validación. El informe desglosa la cobertura del detector y compara ambos modelos sobre las mismas fotos de prueba con rasgos disponibles.

```powershell
python scripts/prepare_dataset.py --dataset fer2013 --source data/fer2013.csv --output data/fer2013.npz
python scripts/train_fer2013.py --data data/fer2013.npz --destination data/fer2013_run
```

## Versión 3 · JAFFE y WSEFEP

La versión 3 añade el Warsaw Set of Emotional Facial Expression Pictures de Olszanowski y colaboradores, con 210 fotografías posadas de 30 personas. Se conservan alegría, tristeza, enfado, sorpresa y neutral. Las etiquetas y los identificadores de persona proceden de la hoja incluida en el archivo oficial. Una foto indicada en la hoja faltaba en el ZIP y se contabilizó como descartada. WSEFEP se ofrece para investigación científica no comercial. Las imágenes, recortes y datos de rasgos quedan fuera del paquete. [Página de los autores y condiciones](https://www.rcbbsb.com/wsefep) · [Archivo del proyecto en OSF](https://osf.io/7xgz9/).

La semilla 431 reserva ocho personas de WSEFEP para el test de esta versión. Se conservan también los dos sujetos históricos de JAFFE. Las otras 30 personas aportan 231 fotos de desarrollo. En cuatro particiones de validación se mantienen juntas todas las fotos y variantes de cada persona. Solo las fotos de ajuste reciben copias reflejadas y cambios de luz y nitidez. Se comparan movimientos faciales solos o combinados con geometría, kernel lineal o RBF, cuatro valores de C y entrenamiento con o sin variantes. Venció la SVM lineal con C=0,1 y 128 rasgos, con F1 macro de validación 0,825. El ajuste final usa 693 ejemplos contando las copias.

| Conjunto reservado | V2 acierto / F1 macro | V3 acierto / F1 macro |
|---|---:|---:|
| WSEFEP, 40 fotos nuevas | 22/40 (55,0 %) / 0,517 | 33/40 (82,5 %) / 0,826 |
| JAFFE, 30 fotos históricas | 20/30 (66,7 %) / 0,667 | 20/30 (66,7 %) / 0,654 |

Los ocho sujetos WSEFEP se fijaron antes de entrenar y quedaron fuera de selección y ajuste. Las 40 fotos se evaluaron con ambos modelos. El aumento de acierto se observa en este conjunto de fotos posadas de laboratorio. La webcam presenta otras condiciones. Los detalles por clase quedan en `reports/metrics.json` y el modelo previo se conserva en `models/emotion_v2.joblib`.

Para reproducir desde la carpeta del proyecto, prepara JAFFE como en los comandos de abajo, descarga [WSEFEP v1.0.1 de OSF](https://osf.io/download/2g3ej/) en la carpeta de trabajo y ejecuta:

```powershell
python scripts/train_v3.py --jaffe ../../work/jaffe --warsaw ../../work/wsefep.zip --features ../../work/features_v3.npz --destination ../../work/v3_reproducido
```

El script verifica el SHA-256 de WSEFEP al extraer los rasgos. Utiliza la V2 incluida como referencia y escribe los nuevos artefactos en `--destination`.

## Versión 2: cambio real de representación y entrenamiento

En la V2 entrenamos **una SVM propia sobre 128 rasgos faciales**: 52 coeficientes de movimiento facial de MediaPipe y 76 coordenadas de ojos, cejas y labios. Estas coordenadas se alinean al eje ocular y se normalizan por la distancia entre ojos; así se eliminan traslación, escala y giro en el plano. Se corrige la relación de aspecto de la imagen antes de calcular la geometría.

MediaPipe es un extractor preentrenado de landmarks y movimientos, **no el clasificador final de las cinco expresiones**. Las etiquetas happy, sad, angry, surprise y neutral se aprenden con nuestra SVM y JAFFE. No se usa DeepFace.analyze ni un modelo de clasificación de emociones ya entrenado para sustituir el entrenamiento propio. El enfoque corresponde a extraer características y entrenar un clasificador, como en los ejemplos de la práctica. El HOG + SVM original queda conservado como comparación.

Referencia del extractor: [documentación oficial Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python). Los coeficientes se ordenan alfabéticamente por category_name tanto al entrenar como al ejecutar. Las 76 coordenadas y su normalización están definidas en geometry_vector(), compartida entre ambos procesos. Se usa la imagen original completa al extraer rasgos del dataset; en webcam se reutiliza la misma observación MediaPipe del filtro, evitando una segunda inferencia.

## Resultados reales y comparación

| Método | Aciertos sobre benchmark histórico | Accuracy | F1 macro |
|---|---:|---:|---:|
| V1 HOG + SVM | 14/30 | 46,67 % | 0,4494 |
| V2 movimientos + geometría + SVM | 20/30 | 66,67 % | 0,6667 |

| Clase | F1 de V2 | Soporte |
|---|---:|---:|
| Alegría | 0,8000 | 6 |
| Tristeza | 0,5455 | 6 |
| Enfado | 0,6154 | 6 |
| Sorpresa | 0,6667 | 6 |
| Neutral | 0,7059 | 6 |

La validación agrupada de desarrollo obtiene F1 macro medio **0,8006**, con cuatro folds de 0,8626, 0,8031, 0,7152 y 0,8216. Es una métrica de selección, no una estimación independiente de rendimiento final.

**Honestidad de la evaluación:** las 30 imágenes de KL y UY ya se habían consultado en V1 y en un experimento inicial V2 de solo coeficientes (46,67 %). Por tanto, ahora se denominan **benchmark histórico**, no test ciego nuevo. La versión combinada se eligió exclusivamente mediante validación agrupada sobre las otras ocho personas; se fijó antes de volver a evaluar ese benchmark. Tras obtener el 66,67 % no se siguió ajustando el modelo. Una medición independiente requeriría nuevos sujetos o un dataset externo nunca utilizado.

Los informes de V2 están en `reports/baseline_v2/` y `reports/development_selection.json`. Los resultados de V1 se conservan en `reports/baseline_v1/` y su modelo en `models/emotion_baseline.joblib`.

## Dataset y condiciones de uso

**JAFFE v2**, de Michael Lyons, Miyuki Kamachi y Jiro Gyoba. La [página original](https://www.kasrl.org/jaffe.html) enlaza al [registro oficial de Zenodo](https://zenodo.org/records/14974867). Son 213 imágenes TIFF de 256×256, diez participantes, siete expresiones posadas. Conservamos 152 imágenes de cinco clases: HA→happy, SA→sad, AN→angry, SU→surprise y NE→neutral. Se excluyen DI y FE. LeBron solo es un recurso visual: sus fotografías no entrenan el modelo.

JAFFE permite investigación científica no comercial y prohíbe redistribuir imágenes y publicarlas en redes. **No se incluyen imágenes, recortes ni archivos de rasgos del dataset en la entrega.** Las demostraciones publicables deben usar tu propia webcam. Consulta todas las condiciones del registro y README_FIRST.txt antes de usarlo. El ZIP oficial pesa aproximadamente 12,3 MB y tiene MD5 fe13f3302eb9968ef04367456f665436, verificado por el preparador.

Referencias solicitadas por los autores: Lyons, Kamachi y Gyoba (2020), [Coding Facial Expressions with Gabor Wavelets](https://arxiv.org/abs/2009.05938); Lyons (2021), [“Excavating AI” Re-excavated](https://arxiv.org/abs/2107.13998).

## Selección sin fugas de sujetos

Se conserva la división inicial por semilla 42: seis personas de entrenamiento, dos de validación y dos del benchmark histórico. V2 combina los ocho sujetos de desarrollo (122 imágenes) para **GroupKFold de cuatro folds**, sin una misma persona a ambos lados de un fold. El StandardScaler se ajusta desde cero únicamente con la parte de entrenamiento de cada fold. Se comparan tres representaciones (52 coeficientes, 76 coordenadas y combinación), SVM lineal/RBF y C entre 0,01, 0,1, 1, 10 y 100, eligiendo por F1 macro medio.

La combinación, kernel lineal y C=0,01 ganan con 0,8006. Se ajusta esa configuración final usando las 122 imágenes de los ocho sujetos de desarrollo. KL y UY quedan fuera de ese ajuste. No se entrena en las 30 imágenes históricas. Todos los 152 rostros se detectaron con MediaPipe, sin descartes.

El clasificador no expone porcentajes de confianza porque sus salidas no están calibradas. El archivo joblib solo debe cargarse si procede de este entrenamiento local.

## Preparar JAFFE y reproducir V2

Desde la raíz del proyecto y con requirements.txt instalado:

```powershell
python scripts/prepare_dataset.py --dataset jaffe --source ../../work/jaffe --download-jaffe --output ../../work/jaffe_prepared.npz
python scripts/train_v2.py --source ../../work/jaffe --features ../../work/jaffe_geometric.npz --select-only
python scripts/train_v2.py --source ../../work/jaffe --features ../../work/jaffe_geometric.npz
```

El primer comando verifica y descarga la fuente. Si ya está descomprimida, omite `--download-jaffe`. El segundo realiza selección sin evaluar el benchmark. **Ejecuta el tercero solo en una copia del proyecto** porque ese script histórico escribe en las rutas del modelo activo y los informes. Los rasgos privados se guardan fuera del proyecto. El paquete incluye el modelo activo y los modelos MediaPipe.

Para reproducir únicamente HOG V1 sin reemplazar el modelo activo:

```powershell
python scripts/train.py --data ../../work/jaffe_prepared.npz --model models/emotion_hog_reproduced.joblib --report-dir reports/hog_reproduced
```

El preparador también admite el CSV original FER2013 (emotion,pixels,Usage) obtenido desde su [competición oficial](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), previa aceptación de condiciones. No se descargan espejos sin verificar sus derechos. Su división Usage no proporciona identidades; no garantiza separación por persona. Ese soporte alimenta train.py, no el extractor V2, que espera imágenes JAFFE originales.

## Limitaciones y siguiente paso útil

JAFFE sigue siendo muy pequeño, de condiciones controladas y participantes poco variados. No se arregla esa limitación aumentando épocas ni con un porcentaje elevado de un test mezclado entre personas. Tristeza y enfado todavía se confunden. Las etiquetas son expresiones posadas, no sentimientos internos conocidos. Hay que comprobar distancia, luz, inclinación, gafas y expresividad con personas nuevas.

La mejora actual es real dentro del benchmark disponible, pero no convierte la aplicación en un sistema fiable de lectura emocional. El siguiente paso es un dataset más variado con procedencia y permisos verificados y un nuevo test independiente. Después tendría sentido comparar una CNN pequeña con este clasificador; no se ha entrenado una CNN ni se promete que mejore por sí sola.
