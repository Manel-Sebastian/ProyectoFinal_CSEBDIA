# ProyectoFinal_CSEBDIA
Proyecto final de curso sobre la implementación de red neuronal para filtrar si en una foto aparecen dedos o no

# Librerías del Proyecto

Estas son todas las librerias necesarias del proyecto. PAra descargarlas todas a la vez necesitas hacer el comando("pip install -r requirements.txt")

## Librerías Estándar de Python

| Librería | Para qué sirve |
|----------|---------------|
| `sys` | Interactuar con el intérprete de Python. Acceder a argumentos de la línea de comandos, salir del programa o modificar el path. |
| `os` | Interactuar con el sistema operativo. Manejar rutas, crear carpetas, listar archivos y leer variables de entorno. |
| `random` | Generar números y selecciones aleatorias. Útil para barajar datos, inicializar semillas o hacer muestreos. |
| `pathlib` | Manejo moderno de rutas de archivos y directorios de forma orientada a objetos. Alternativa más limpia a `os.path`. |
| `pprint` | *Pretty print*. Imprime estructuras de datos complejas (listas, diccionarios) de forma legible y formateada. |

---

## Librerías Externas
### Visualización

| Librería | Para qué sirve |
|----------|---------------|
| `matplotlib` | Crear gráficos, histogramas, curvas de entrenamiento e imágenes. Base de la visualización en Python. |
| `seaborn` | Visualización estadística construida sobre matplotlib. Gráficos más estéticos y complejos con menos código. |

### Datos y Números

| Librería | Para qué sirve |
|----------|---------------|
| `numpy` | Computación numérica con arrays multidimensionales. Base de casi toda la ciencia de datos en Python. |
| `pandas` | Manipulación y análisis de datos en tablas (DataFrames). Ideal para cargar CSVs, filtrar, agrupar y hacer estadísticas. |

### Deep Learning (PyTorch)

| Librería | Para qué sirve |
|----------|---------------|
| `torch` | Framework principal de Deep Learning. Define y entrena redes neuronales con soporte GPU mediante CUDA. |
| `torch.nn` | Construir redes neuronales. Contiene capas (`Linear`, `Conv2d`...), funciones de activación y de pérdida. |
| `torch.optim` | Algoritmos de optimización como Adam o SGD para actualizar los pesos de la red durante el entrenamiento. |
| `torch.optim.lr_scheduler` | Ajusta automáticamente el learning rate cuando la métrica deja de mejorar, evitando estancamientos. |
| `torch.utils.data` | `Dataset`: define cómo cargar tus datos. `DataLoader`: los sirve en batches con shuffling y paralelismo. |
| `torchvision.transforms` | Transformaciones para imágenes: redimensionar, normalizar, rotar o aplicar data augmentation. |

### Imágenes

| Librería | Para qué sirve |
|----------|---------------|
| `Pillow (PIL)` | Abrir, manipular y guardar imágenes en múltiples formatos (JPG, PNG, etc.). Base del procesamiento de imágenes. |

### Machine Learning (Scikit-learn)

| Librería | Para qué sirve |
|----------|---------------|
| `train_test_split` | Divide el dataset en conjuntos de entrenamiento y test de forma aleatoria y estratificada. |
| `compute_class_weight` | Calcula pesos por clase para compensar datasets desbalanceados durante el entrenamiento. |
| `accuracy_score` | Porcentaje de predicciones correctas sobre el total. |
| `classification_report` | Resumen completo de precision, recall y f1-score por cada clase. |
| `confusion_matrix` | Matriz que muestra aciertos y errores del modelo por clase. |
| `f1_score` | Media armónica entre precision y recall. Útil cuando las clases están desbalanceadas. |
| `precision_score` | De todos los que predijo como positivos, cuántos realmente lo eran. |
| `recall_score` | De todos los que eran positivos, cuántos consiguió detectar el modelo. |

### Utilidades

| Librería | Para qué sirve |
|----------|---------------|
| `tqdm` | Muestra barras de progreso en bucles y entrenamientos. Permite ver el avance epoch a epoch. |
