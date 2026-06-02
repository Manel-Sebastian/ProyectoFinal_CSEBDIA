# 🩻 Asistente IA de Posicionamiento Radiológico

Sistema de inteligencia artificial para la validación automática del posicionamiento de dedos en radiografías. Combina modelos de visión artificial entrenados localmente (PyTorch) con la IA multimodal de Google Gemini, todo accesible desde una interfaz web sencilla construida con Streamlit.

---

## 📋 Índice

1. [¿Qué hace este proyecto?](#qué-hace-este-proyecto)
2. [Arquitectura general](#arquitectura-general)
3. [Estructura de archivos](#estructura-de-archivos)
4. [Dataset](#dataset)
5. [Modelos de IA](#modelos-de-ia)
6. [Notebooks de entrenamiento](#notebooks-de-entrenamiento)
7. [Módulo de reentrenamiento — retrain.py](#módulo-de-reentrenamiento--retrainpy)
8. [Aplicación web — app.py](#aplicación-web--apppy)
9. [Instalación y puesta en marcha](#instalación-y-puesta-en-marcha)
10. [Uso de la aplicación](#uso-de-la-aplicación)
11. [Sistema de métricas](#sistema-de-métricas)
12. [Cómo añadir un nuevo modelo](#cómo-añadir-un-nuevo-modelo)
13. [Tecnologías utilizadas](#tecnologías-utilizadas)

---

## ¿Qué hace este proyecto?

Cuando un técnico hace una radiografía de un dedo, la posición de la mano debe ser muy precisa para que la imagen sea diagnósticamente útil. Este sistema automatiza la comprobación: el técnico sube la imagen y la IA le dice en segundos si la postura es **correcta** o **incorrecta**.

El proyecto resuelve un problema de **clasificación binaria de imágenes médicas**:

```
Entrada: imagen de radiografía (JPG/PNG)
Salida:  "si" (postura correcta) / "no" (postura incorrecta)
```

Adicionalmente, el sistema permite **reentrenar los modelos desde la propia interfaz**, sin necesidad de conocimientos técnicos, y guarda automáticamente solo el modelo que mejore al anterior.

---

## Arquitectura general

```
┌─────────────────────────────────────────────────────────────┐
│                     STREAMLIT (app.py)                      │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │
│  │ Subir foto  │  │ Usar cámara │  │  Actualizar la IA  │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────┬──────────┘  │
│         │                │                    │             │
│         └────────┬───────┘                    │             │
│                  ▼                            ▼             │
│         ┌────────────────┐          ┌─────────────────┐    │
│         │ Selector de IA │          │   retrain.py    │    │
│         └───────┬────────┘          └────────┬────────┘    │
│                 │                             │             │
│        ┌────────┴────────┐                   │             │
│        ▼                 ▼                   ▼             │
│  ┌──────────┐    ┌──────────────┐   ┌─────────────────┐   │
│  │  Gemini  │    │  PyTorch .pt │   │ Entrenamiento   │   │
│  │  (nube)  │    │  (local)     │   │ ResNet18        │   │
│  └──────────┘    └──────────────┘   │ MobileNetV2     │   │
│                                     └────────┬────────┘   │
│                                              │             │
│                                     ┌────────▼────────┐   │
│                                     │ metrics_history │   │
│                                     │     .json       │   │
│                                     └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Estructura de archivos

```
/home/carlos/Escritorio/a/
│
├── app.py                  # Aplicación web Streamlit (interfaz de usuario)
├── retrain.py              # Módulo de reentrenamiento automático
├── ResNet18.ipynb          # Notebook de experimentación con ResNet18
├── MobileNet.ipynb         # Notebook de experimentación con MobileNetV2
└── README.md               # Este documento
```

```
/home/carlos/dataset/imagenes/      # Dataset (ruta configurable)
└── dedo/
    ├── si/                         # Radiografías con postura CORRECTA
    │   ├── imagen_001.jpg
    │   └── ...
    └── no/                         # Radiografías con postura INCORRECTA
        ├── imagen_001.jpg
        └── ...
```

```
/home/carlos/modelos/               # Modelos entrenados (ruta configurable)
└── dedo/
    ├── resnet18_finetuning.pt      # Mejor modelo ResNet18 guardado
    └── mobilenet_augmentation.pt   # Mejor modelo MobileNetV2 guardado
    metrics_history.json            # Historial completo de entrenamientos
```

---

## Dataset

### Estructura
El sistema espera imágenes organizadas en carpetas `si/` y `no/` dentro de una carpeta que identifica la **parte del cuerpo**. Esta estructura permite escalar fácilmente a otras partes anatómicas en el futuro.

### Características del dataset actual
| Parámetro | Valor |
|---|---|
| Parte del cuerpo | Dedo |
| Imágenes clase "si" | 44 |
| Imágenes clase "no" | 37 |
| Total | 81 imágenes |
| Formatos | JPG, JPEG, PNG, BMP |

### División train/val/test
El módulo `retrain.py` divide automáticamente el dataset en cada entrenamiento:

```
Total (81)
  ├── Train  (64%) — 51 imágenes — el modelo aprende con estas
  ├── Val    (16%) — 13 imágenes — controla si el modelo mejora o empeora durante el entrenamiento
  └── Test   (20%) — 17 imágenes — evaluación final e imparcial del modelo
```

La división es **estratificada**: mantiene la misma proporción de `si`/`no` en cada split para que la evaluación sea justa.

---

## Modelos de IA

El proyecto entrena y compara dos arquitecturas de redes neuronales convolucionales (CNN) preentrenadas en ImageNet y adaptadas mediante **transfer learning** al problema específico de las radiografías.

### ResNet18 — Fine-Tuning

**¿Qué es?** ResNet18 es una red neuronal de 18 capas diseñada por Microsoft Research en 2015. Su innovación principal son las **conexiones residuales** (shortcuts) que permiten entrenar redes más profundas sin que el gradiente desaparezca.

```
Parámetros totales:   11.177.538
Parámetros entrenados: 10.494.466 (93,9%)
Capas descongeladas:  layer3 + layer4 + fc
```

**Estrategia de fine-tuning:**
```python
# Se congela todo el backbone
for param in model.parameters():
    param.requires_grad = False

# Se descongelan solo las capas profundas
for name, param in model.named_parameters():
    if name.startswith(("layer3", "layer4", "fc")):
        param.requires_grad = True

# Cabeza personalizada con dropout para evitar overfitting
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(512, 2)   # 2 clases: si / no
)
```

**Hiperparámetros:**
| Parámetro | Valor | Motivo |
|---|---|---|
| LR backbone | 1e-6 | Muy bajo para no destruir pesos preentrenados |
| LR cabeza | 1e-5 | Algo más alto para adaptar la capa final |
| Épocas máx. | 20 | Con early stopping |
| Optimizador | Adam | Convergencia estable |
| Scheduler | CosineAnnealing | Reduce el LR suavemente al final |

---

### MobileNetV2 — Con Data Augmentation

**¿Qué es?** MobileNetV2 es una arquitectura de Google diseñada para ser eficiente en dispositivos con recursos limitados. Usa **bloques residuales invertidos** y **convoluciones depthwise separables** que reducen los parámetros a una fracción de los de VGG16 o ResNet manteniendo una precisión comparable.

```
Parámetros totales:    2.226.434
Parámetros entrenados: 1.683.906 (75,6%)
Capas descongeladas:   features.14-18 + classifier
```

**Data Augmentation con Albumentations:**

Con solo 81 imágenes, el riesgo de overfitting es alto. La augmentation genera variaciones artificiales durante el entrenamiento para que el modelo generalice mejor:

```python
aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),         # Simula manos derechas e izquierdas
    A.Rotate(limit=20, p=0.6),        # Variaciones de orientación reales
    A.RandomResizedCrop(..., p=0.5),  # Distintas distancias a la cámara
    A.ColorJitter(..., p=0.6),        # Diferentes condiciones de iluminación
    A.GaussNoise(..., p=0.3),         # Ruido de cámaras de baja calidad
    A.GaussianBlur(..., p=0.2),       # Imágenes ligeramente desenfocadas
])
```

**Hiperparámetros:**
| Parámetro | Valor |
|---|---|
| LR backbone | 1e-5 |
| LR cabeza | 1e-3 |
| Épocas máx. | 15 |
| Optimizador | Adam + CosineAnnealing |

---

### Comparativa de arquitecturas

| Característica | ResNet18 | MobileNetV2 |
|---|---|---|
| Parámetros | 11,2 M | 2,2 M |
| Tamaño disco | ~45 MB | ~9 MB |
| Velocidad CPU | Media | Rápida |
| Augmentation | Básica (torchvision) | Avanzada (albumentations) |
| Mejor para | Datasets pequeños | Edge / producción |

---

## Notebooks de entrenamiento

### ResNet18.ipynb
Notebook de experimentación completo que compara **tres estrategias** de uso de una red preentrenada:

| Estrategia | Parámetros entrenables | Qué hace |
|---|---|---|
| **Zero-Shot** | 0 | Evalúa la red tal como viene de ImageNet, sin entrenar nada |
| **Feature Extraction** | 1.026 | Solo entrena la capa final; el backbone actúa como extractor fijo |
| **Fine-Tuning** | 10.494.466 | Desbloquea las últimas capas para adaptar las representaciones al dominio |

**Conclusión del notebook:** El Fine-Tuning supera a las otras estrategias porque permite que la red reescriba sus representaciones internas para detectar estructuras específicas de radiografías de dedos.

### MobileNet.ipynb
Compara el mismo modelo entrenado con y sin data augmentation:

| Variante | Accuracy | F1 (si) |
|---|---|---|
| Baseline (sin augmentation) | 53,3% | 53,3% |
| Con Augmentation | 60,0% | 62,5% |

**Conclusión:** La augmentation mejora especialmente el recall, es decir, el modelo detecta mejor las posturas correctas sin dejar escapar ninguna.

---

## Módulo de reentrenamiento — retrain.py

Este módulo contiene toda la lógica de entrenamiento en forma de funciones reutilizables. Puede ejecutarse desde la terminal o invocarse desde Streamlit.

### Componentes principales

#### `MODEL_REGISTRY` — Registro de modelos
Diccionario central donde se definen todos los modelos disponibles. Para añadir uno nuevo, basta con añadir una entrada aquí.

```python
MODEL_REGISTRY = {
    "resnet18_finetuning": {
        "build_fn":    build_resnet18_finetuning,  # función que construye la arquitectura
        "epochs":      20,
        "lr_backbone": 1e-6,
        "lr_head":     1e-5,
        "filename":    "resnet18_finetuning.pt",
        "description": "ResNet18 fine-tuning (layer3+layer4+fc)",
        ...
    },
    "mobilenet_augmentation": { ... },
}
```

#### `train_model()` — Loop de entrenamiento
Implementa el ciclo estándar de entrenamiento con:
- **Early stopping** (patience=7): para automáticamente si el modelo no mejora en 7 épocas consecutivas, evitando overfitting
- **Guardado del mejor checkpoint**: solo guarda los pesos cuando la validación mejora
- **Scheduler coseno**: reduce el learning rate de forma suave al final del entrenamiento

```
Para cada época:
    ├── Fase train:  forward → loss → backward → actualizar pesos
    ├── Fase val:    forward → calcular métricas (sin actualizar pesos)
    └── Si val_acc mejora → guardar pesos como "mejor modelo"
```

#### Sistema de "mejor modelo"
Antes de sobreescribir el archivo `.pt` en disco, el sistema consulta el historial de métricas y **solo guarda si el nuevo modelo supera al mejor registrado**:

```python
def _get_best_score(body_part, model_key, models_dir):
    """Consulta el mejor F1 histórico para esta combinación."""
    history = load_metrics_history(models_dir)
    scores  = [m["f1_si"] for m in history
               if m["body_part"] == body_part
               and m["model"]    == model_key
               and m.get("saved_as_best", False)]
    return max(scores) if scores else -1.0

# En train_one_model:
if nuevo_f1 > mejor_f1_historico:
    torch.save(model.state_dict(), model_path)  # ✅ Sobreescribe
    metrics["saved_as_best"] = True
else:
    metrics["saved_as_best"] = False             # ❌ Conserva el anterior
```

#### `retrain_all_models()` — Función principal
Orquesta todo el proceso con soporte para callbacks en tiempo real:

```python
retrain_all_models(
    dataset_dir  = "/home/carlos/dataset/imagenes",  # ruta al dataset
    models_dir   = "/home/carlos/modelos",           # donde guardar los .pt
    model_keys   = ["resnet18_finetuning"],           # None = todos
    log_fn       = print,           # callback para mensajes en tiempo real
    progress_fn  = lambda v: ...,   # callback de progreso 0.0–1.0
)
```

El sistema detecta automáticamente las partes del cuerpo disponibles escaneando la estructura de carpetas, y soporta dos modos:
- **Carpeta directa:** `dataset_dir/si/` y `dataset_dir/no/` → una sola parte del cuerpo
- **Carpeta padre:** `dataset_dir/dedo/si/`, `dataset_dir/rodilla/si/` → múltiples partes

#### `metrics_history.json` — Persistencia
Cada entrenamiento añade una entrada al historial sin borrar las anteriores:

```json
[
  {
    "model":           "resnet18_finetuning",
    "body_part":       "dedo",
    "timestamp":       "2026-06-01T19:30:00",
    "accuracy":        0.7333,
    "balanced_accuracy": 0.7143,
    "precision_si":    0.7778,
    "recall_si":       0.8750,
    "f1_si":           0.8235,
    "precision_no":    0.6667,
    "recall_no":       0.5000,
    "f1_no":           0.5714,
    "val_loss_best":   0.4821,
    "train_loss_best": 0.3102,
    "epochs_trained":  14,
    "test_samples":    17,
    "saved_as_best":   true,
    "best_f1_si_prev": 0.7500
  },
  ...
]
```

---

## Aplicación web — app.py

Interfaz construida con Streamlit que expone toda la funcionalidad al usuario final sin requerir ningún conocimiento técnico.

### Pestañas

#### 📁 Subir archivo / 📸 Usar cámara
El usuario carga una imagen. Inmediatamente aparece:
1. **Vista previa** de la radiografía
2. **Selector de IA** (radio button) con todos los modelos disponibles
3. **Botón de análisis** que devuelve el resultado

El selector se construye dinámicamente escaneando la carpeta de modelos configurada en el sidebar:
```python
def descubrir_modelos(models_dir):
    for pt in models_dir.rglob("*.pt"):
        etiqueta = f"🧠 ResNet18 — {parte}"  # nombre amigable
        encontrados[etiqueta] = str(pt)       # ruta real del archivo
```

Los modelos se cargan con `@st.cache_resource` para que permanezcan en memoria entre análisis consecutivos, evitando recargar el archivo `.pt` cada vez.

#### 📊 Resultados de los modelos
Muestra las métricas de rendimiento de los modelos. La tabla técnica completa está oculta en un expander para no abrumar al usuario.

#### 🤖 Actualizar la IA
Panel de reentrenamiento con:

- **Configuración avanzada** (colapsada): rutas del dataset/modelos y selector de qué modelos reentrenar
- **Botón principal**: "Iniciar actualización de la IA"
- **Durante el entrenamiento**: barra de progreso + log en vivo + contador de tiempo
- **Tras el entrenamiento**: mensajes claros por modelo indicando si mejoró o no
- **Historial**: tabla de todos los entrenamientos previos con columna "¿Mejoró?"

### Flujo de inferencia
```python
def render_analisis(imagen, key_suffix):
    # 1. Selector de modelo (radio)
    etiqueta    = st.radio("¿Qué IA quieres usar?", opciones_modelo.keys())
    modelo_ruta = opciones_modelo[etiqueta]

    # 2. Botón
    if st.button("Analizar"):

        # 3. Clasificación según el modelo elegido
        if modelo_ruta == "gemini":
            resultado = clasificar_con_gemini(imagen)    # llama a la API de Google
        else:
            resultado = clasificar_con_pytorch(imagen, modelo_ruta)  # inferencia local

        # 4. Resultado
        if resultado == "si":
            st.success("✅ POSTURA CORRECTA")
        elif resultado == "no":
            st.error("❌ POSTURA INCORRECTA")
```

### Sesión y estado
Streamlit re-ejecuta el script completo en cada interacción del usuario. Para preservar los resultados del entrenamiento entre reruns se usa `st.session_state`:

```python
# Al terminar el entrenamiento:
st.session_state["retrain_results"] = results
st.rerun()  # recarga la app

# Al renderizar:
results = st.session_state.get("retrain_results")  # recupera los resultados
```

---

## Instalación y puesta en marcha

### Requisitos
- Python 3.10 o 3.11
- Git (opcional, para clonar)
- ~4 GB de RAM (para cargar los modelos)

---

### 1. Entrar en la carpeta del proyecto

```bash
cd AppReentrenoModelos
```

---

### 2. Crear y activar el entorno virtual

**Con `venv` (recomendado):**

```bash
# Crear entorno
python -m venv venv

# Activar (Windows PowerShell/CMD)
venv\Scripts\activate

# Activar (Windows Git Bash / Linux / macOS)
source venv/bin/activate
```

**Con Conda:**

```bash
conda create -n radiologia python=3.11 -y
conda activate radiologia
```

---

### 3. Instalar dependencias

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

> **Nota sobre PyTorch:** Si vas a usar GPU NVIDIA, instala PyTorch manualmente con el índice CUDA correspondiente antes del paso anterior. Ejemplo para CUDA 12.1:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> Si usas CPU, el `requirements.txt` ya instalará la versión CPU por defecto.

---

### 4. Configurar variables de entorno

Crea un archivo `.env` en la misma carpeta que `app.py` con el siguiente contenido:

```bash
# Copia este bloque y pégalo en un archivo llamado .env
echo GEMINI_API_KEY=tu_clave_aqui > .env
echo MULTIMODAL_MODEL_GEMINI=gemini-2.5-flash >> .env
```

O créalo a mano:

```
GEMINI_API_KEY=tu_clave_aqui
MULTIMODAL_MODEL_GEMINI=gemini-2.5-flash
```

> Sin esta clave, Gemini no funcionará, pero los modelos locales (ResNet18 / MobileNet) seguirán disponibles si los has entrenado previamente.

---

### 5. Lanzar la aplicación

```bash
streamlit run app.py
```

La aplicación se abre automáticamente en **http://localhost:8501**

---

### Resumen de comandos (copiar y pegar de una sola vez)

**Windows (PowerShell):**
```powershell
cd AppReentrenoModelos
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$null > .env
Add-Content .env "GEMINI_API_KEY=tu_clave_aqui"
Add-Content .env "MULTIMODAL_MODEL_GEMINI=gemini-2.5-flash"
streamlit run app.py
```

**Windows (CMD):**
```cmd
cd AppReentrenoModelos
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
echo GEMINI_API_KEY=tu_clave_aqui > .env
echo MULTIMODAL_MODEL_GEMINI=gemini-2.5-flash >> .env
streamlit run app.py
```

**Linux / macOS / Git Bash:**
```bash
cd AppReentrenoModelos
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=tu_clave_aqui" > .env
echo "MULTIMODAL_MODEL_GEMINI=gemini-2.5-flash" >> .env
streamlit run app.py
```

---

## Uso de la aplicación

### Analizar una radiografía
1. Abre la pestaña **"Subir archivo"** o **"Usar cámara"**
2. Carga la imagen de la radiografía
3. Selecciona la IA que quieres usar en el selector
4. Pulsa **"Analizar la radiografía"**
5. El resultado aparece en segundos

### Actualizar/reentrenar los modelos
1. Abre la pestaña **"Actualizar la IA"**
2. Si quieres personalizar, abre **"Configuración avanzada"** y ajusta las rutas
3. Pulsa **"Iniciar actualización de la IA"**
4. Observa el progreso en tiempo real
5. Al terminar, aparecen mensajes indicando si cada modelo mejoró o no

---

## Sistema de métricas

### Métricas registradas por entrenamiento

| Métrica | Qué mide |
|---|---|
| **Accuracy** | Porcentaje de imágenes clasificadas correctamente |
| **Balanced Accuracy** | Accuracy corregido cuando hay más imágenes de una clase que de otra |
| **Precision (si)** | De las que dijo "correctas", ¿cuántas realmente lo eran? |
| **Recall (si)** | De las que eran correctas, ¿cuántas encontró? |
| **F1 (si)** | Media armónica entre Precision y Recall — métrica principal |
| **Val Loss** | Error en el conjunto de validación durante el entrenamiento |
| **Épocas entrenadas** | Cuántas iteraciones completas realizó antes de parar |

### ¿Por qué F1 como métrica principal?
En un problema médico, tanto los falsos positivos (decir "correcto" cuando no lo es) como los falsos negativos (no detectar una mala postura) son importantes. El **F1-Score** penaliza ambos tipos de error por igual, siendo más fiable que el Accuracy cuando el dataset no está perfectamente balanceado.

---

## Cómo añadir un nuevo modelo

Gracias al `MODEL_REGISTRY`, añadir un nuevo modelo solo requiere **tres pasos**:

**1. Implementar la función constructora en `retrain.py`:**
```python
def build_mi_nuevo_modelo(num_classes=2):
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # ... adaptar la cabeza ...
    return model.to(device)
```

**2. Registrarlo en `MODEL_REGISTRY`:**
```python
MODEL_REGISTRY["efficientnet_b0"] = {
    "build_fn":    build_mi_nuevo_modelo,
    "epochs":      15,
    "lr_backbone": 1e-5,
    "lr_head":     1e-3,
    "use_augmentation": True,
    "head_prefixes": ("classifier",),
    "filename":    "efficientnet_b0.pt",
    "description": "EfficientNet-B0 con augmentation",
}
```

**3. Reentrenar desde la app.**

El nuevo modelo aparecerá automáticamente en el selector de la interfaz y en el multiselect del panel de reentrenamiento.

---

## Tecnologías utilizadas

| Tecnología | Versión | Uso |
|---|---|---|
| **Python** | 3.11 | Lenguaje principal |
| **PyTorch** | 2.11 | Entrenamiento e inferencia de redes neuronales |
| **Torchvision** | — | Arquitecturas preentrenadas (ResNet18, MobileNetV2) |
| **Albumentations** | 2.0 | Pipeline de data augmentation avanzado |
| **Streamlit** | 1.58 | Interfaz web interactiva |
| **scikit-learn** | 1.8 | Métricas de evaluación |
| **Google Gemini** | 2.5 Flash | Modelo multimodal en la nube (opcional) |
| **Pandas** | 3.0 | Manipulación de datos y tablas |
| **Pillow** | 12.2 | Carga y procesado de imágenes |
| **Conda** | — | Gestión del entorno Python |

---

## Autores

Proyecto desarrollado como parte del curso de Inteligencia Artificial.
