# Práctica 3 - Visión por Computador

## Autoría

**Grupo 33**

- Álvaro García Suárez ([Github](https://github.com/agsuarezz))[text](../../../../../Downloads/README.md)
- Marcos González Gómez ([Github](https://github.com/mgz00))

---

## Descripción

En esta práctica se han aplicado técnicas de procesamiento de imagen con OpenCV para detectar objetos, obtener información sobre ellos y clasificarlos. El trabajo se divide en dos tareas:

1. **Detección y conteo de monedas de euro**, tanto en una imagen estática como mediante webcam, identificando su valor y calculando la cantidad total.
2. **Detección y clasificación de microplásticos**, distinguiendo entre fragmentos (FRA), pellets (PEL) y alquitrán (TAR) a partir de características visuales y métodos de aprendizaje automático tradicionales.

---

## Trabajo realizado

### 1. Detección, clasificación y suma de monedas

Se parte de la imagen `Monedas.jpg`. Primero se convierte a escala de grises y se utiliza una umbralización automática mediante **Otsu** para separar las monedas del fondo. A continuación, se extraen los contornos y se filtran según su área.

Para estimar el tamaño real de cada moneda se utiliza como referencia la moneda de mayor diámetro, considerada de **2 €**, y se calcula la relación entre píxeles y milímetros. Con esta escala se estima el diámetro del resto de las monedas.

La clasificación combina el **diámetro estimado** con información de **color en el espacio HSV**, diferenciando monedas cobrizas, doradas y bimetálicas. Después se asigna el valor correspondiente y se suman los importes detectados.

#### Resultado en imagen estática

<!-- Añadir captura con las monedas identificadas y el importe total. -->

![Detección de monedas y suma total](images/resultado_monedas.png)

### 2. Demostrador en vivo: detección de monedas con webcam

Se ha añadido una versión en tiempo real que captura fotogramas de la webcam y busca monedas circulares utilizando la **transformada de Hough** (`cv2.HoughCircles`).

El sistema permite calibrar la escala de dos maneras: manualmente, seleccionando una moneda de 1 €, o automáticamente, suponiendo que la moneda detectada de mayor tamaño es de 2 €. Una vez calibrado, identifica las monedas y muestra la suma de su valor.

#### Funcionamiento y controles

- `c`: activar calibración manual; a continuación, hacer clic sobre una moneda de 1 €.
- `a`: calibrar automáticamente a partir de la moneda más grande, considerada de 2 €.
- `+` / `-`: ajustar la sensibilidad de detección.
- `q` o `ESC`: cerrar el demostrador.

La webcam se abre con el índice `1` en el notebook. Si el equipo utiliza otra cámara, se puede cambiar en `cv2.VideoCapture(1)`.

#### Resultado del demostrador

<!-- Añadir captura del detector en funcionamiento. -->

![Detección de monedas con webcam](images/monedas_webcam.png)

<!-- Opcional: añadir GIF del demostrador. -->

![Demostración de monedas en tiempo real](images/demo_monedas.gif)

---

### 3. Segmentación de microplásticos

Para la segunda tarea se utilizan fotografías de entrenamiento correspondientes a tres categorías:

- **FRA:** fragmentos.
- **PEL:** pellets.
- **TAR:** alquitrán.

Las imágenes se segmentan comparando sus niveles de brillo con el fondo. Después se aplican filtros por tamaño y se obtienen contornos para aislar las partículas. Los parámetros del conjunto de entrenamiento se definen a partir de sus propias imágenes.

La imagen de prueba presenta unas condiciones de iluminación distintas, por lo que se utiliza un procedimiento de segmentación adaptado a ella. No se emplean las anotaciones del test para ajustar la detección.

#### Resultado de segmentación

<!-- Añadir una imagen que compare la fotografía original y la máscara de segmentación. -->

![Segmentación de partículas de entrenamiento](images/segmentacion_microplasticos.png)

---

### 4. Extracción de características y clasificación

Una vez obtenidos los contornos, se calculan descriptores que permiten representar cada partícula mediante valores numéricos. Entre ellos se incluyen:

- **Circularidad y compacidad**, relacionadas con la forma.
- **Solidez** y **ocupación del rectángulo delimitador**.
- **Relación entre ejes**, para distinguir formas alargadas o redondeadas.
- **Contraste y color relativos al fondo**, para reducir la influencia de diferencias de iluminación.

Para la clasificación se comparan varios modelos tradicionales: **k-NN, SVM, Random Forest y Extra Trees**. Su selección se realiza mediante validación cruzada estratificada sobre los datos de entrenamiento y utilizando **F1 macro** como criterio. El modelo seleccionado se entrena con las partículas disponibles y se aplica después a la imagen de prueba.

#### Resultado de clasificación

<!-- Añadir captura con las partículas del test marcadas y etiquetadas como FRA, PEL o TAR. -->

![Clasificación de microplásticos](images/clasificacion_microplasticos.png)

---

### 5. Evaluación de los resultados

Para evaluar el sistema se utiliza el archivo `MPs_test_bbs.csv`, que contiene las anotaciones de referencia. Este archivo se lee **únicamente en la fase de evaluación**, después de realizar las predicciones.

Las partículas detectadas se emparejan con las anotaciones mediante la intersección sobre unión (**IoU**). A partir de los emparejamientos se calculan las métricas de clasificación y se muestra una matriz de confusión. También se contabilizan las anotaciones que no se detectaron y las detecciones sin correspondencia.

La evaluación distingue así entre **clasificar correctamente las partículas detectadas** y **detectar todas las partículas presentes**: una buena precisión de clasificación no implica necesariamente una detección completa.

#### Resultados de la evaluación

<!-- Añadir matriz de confusión y, si interesa, una captura de las métricas impresas. -->

![Matriz de confusión](images/matriz_confusion_microplasticos.png)

![Métricas de evaluación](images/metricas_microplasticos.png)

---

## Requisitos

La práctica utiliza Python y las siguientes librerías:

```text
opencv-python
numpy
matplotlib
scikit-learn
```

Pueden instalarse mediante:

```bash
pip install opencv-python numpy matplotlib scikit-learn notebook
```

Para el demostrador de monedas es necesario disponer de una webcam accesible desde OpenCV.

Los archivos de entrada deben estar disponibles en el directorio de trabajo del notebook:

```text
Monedas.jpg
fragment-03-olympus-10-01-2020.JPG
pellet-03-olympus-10-01-2020.JPG
tar-03-olympus-10-01-2020.JPG
MPs_test.jpg
MPs_test_bbs.csv
```

---

## Ejecución

Abrir el notebook unificado:

```bash
jupyter notebook VC_P3_Monedas_y_Microplasticos_COMPLETO.ipynb
```

Ejecutar las celdas en orden. La primera tarea procesa la imagen de monedas y permite ejecutar el demostrador con webcam. La segunda realiza el entrenamiento, la clasificación de microplásticos y la evaluación final.

Para el demostrador de monedas, es necesario mantener activa la ventana de OpenCV durante la interacción.

---

## Conclusiones

La primera tarea permite utilizar **umbrales, contornos, medidas geométricas y análisis de color** para detectar monedas e inferir su valor. La calibración de escala resulta fundamental, ya que los diámetros medidos en píxeles dependen de la distancia a la cámara y del tamaño de la imagen. En la webcam, la transformada de Hough permite extender la detección a fotogramas sucesivos.

En la segunda tarea se combinan técnicas clásicas de segmentación con características geométricas y de apariencia para distinguir tipos de microplásticos sin recurrir a aprendizaje profundo. La detección puede verse afectada por la iluminación, el fondo y la similitud visual entre partículas. Además, la validación sobre partículas extraídas de pocas fotografías puede ofrecer resultados más optimistas que una evaluación con imágenes completamente nuevas.

En conjunto, las dos tareas muestran cómo pasar de la **detección de objetos** a la **extracción de información, clasificación y evaluación de resultados** mediante herramientas de visión por computador.
