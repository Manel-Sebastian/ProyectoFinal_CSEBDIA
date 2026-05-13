# ProyectoFinal_CSEBDIA

Proyecto final de curso sobre clasificación biométrica de posicionamiento de dedo en escáner, desarrollado por los Grupos 2, 3 y 5. El objetivo es determinar automáticamente si un dedo está correctamente colocado sobre la barra de luz del escáner (`SI`) o no (`NO`), comparando tres enfoques distintos: CNNs clásicas con fine-tuning, segmentación médica con MedSAM y clasificación zero-shot con un modelo de lenguaje multimodal.

---

## Estructura del Repositorio

```
ProyectoFinal_CSEBDIA/
├── CNN_desde_0/
│   └── imagenes_procesadas/          ← Dataset (70 imágenes PNG 640×480)
│       ├── train/
│       │   ├── si/   (25 imágenes)
│       │   └── no/   (10 imágenes)
│       └── test/
│           ├── si/   (20 imágenes)
│           └── no/   (15 imágenes)
└── MedSAM_GeminiAPI/
    ├── Dia1/
    │   └── medsam_vit_b.pth          ← Modelo MedSAM preentrenado
    ├── Dia2/
    │   ├── MedSAM_Dia2_notebook.ipynb
    │   └── mascaras_segmentadas_todas/
    └── Dia3/
        ├── ZeroShot_GeminiAPI.ipynb
        └── dia3_Gemini_Sergi.ipynb
```

---

## El Problema

Un escáner biométrico no funciona bien si el dedo no está colocado correctamente. Con un dataset de solo 71 imágenes de radiografías térmicas (infrarrojas), el reto era construir un clasificador fiable — lo que resultó ser cualquier cosa menos trivial.

---

## Enfoque: Cadena de Tres IAs

La arquitectura final emergió de la experimentación a lo largo de tres días de trabajo:

| Paso | Modelo | Rol |
|------|--------|-----|
| 1 | **ResNet18 / MobileNetV2** | CNNs preentrenadas en ImageNet con fine-tuning sobre el dominio del escáner |
| 2 | **MedSAM (ViT-B)** | Segmentación del dedo para aislar la región de interés antes de clasificar |
| 3 | **Gemini 2.0 Flash Lite** | Clasificación zero-shot multimodal vía OpenRouter, sin entrenamiento específico |

---

## Resultados por Modelo

### Grupo 2 — CNNs con Fine-Tuning

| Modelo | Estrategia | Accuracy |
|--------|-----------|----------|
| ResNet18 | Zero-shot (sin entrenar) | 53,33% |
| ResNet18 | Feature Extraction (backbone congelado) | 26,67% |
| ResNet18 | Fine-Tuning (layer3 + layer4 + fc) | 60,00% |
| MobileNetV2 | Sin augmentation | 53,33% |
| MobileNetV2 | Con Albumentations | 53,33% |

> El fine-tuning parcial de ResNet18 fue el único enfoque del Grupo 2 que funcionó. Congelar todo el backbone resultó peor que el azar.

### Grupo 3 — Gemini 2.0 Flash Lite (Zero-Shot)

| Métrica | Valor |
|---------|-------|
| Accuracy global | **70,00%** (49/70 aciertos) |
| Macro F1 | 69,70% |
| Precision (SI) | 68,29% |
| Recall (SI) | 77,78% |

> Gemini superó a todos los modelos entrenados sin ver una sola imagen del dominio.

### MedSAM — Segmentación (Día 2)

| Métrica | Valor |
|---------|-------|
| Imágenes procesadas | 70/70 (100%) |
| Score promedio de confianza | 0.6357 |
| Score mínimo | 0.6064 |
| Desviación estándar | 0.0114 |

> Segmentación completada con éxito. Todos los scores por encima del umbral de 0.60, con distribución muy estable.

---

## Comparativa Final

| Modelo | Accuracy | F1 (SI) | Recall (SI) | Precision (SI) |
|--------|----------|---------|-------------|----------------|
| Gemini Zero-Shot | **70,00%** | **72,73%** | 77,78% | 68,29% |
| ResNet18 Fine-Tuning | 60,00% | 72,73% | **100,00%** | 57,14% |
| ResNet18 Zero-Shot | 53,33% | 53,33% | — | — |
| MobileNetV2 (+Aug) | 53,33% | — | 87,50% | 53,85% |

El F1 de la clase `SI` es idéntico entre Gemini y ResNet18 FT, pero conseguido de formas opuestas: ResNet18 maximiza recall prediciendo casi todo como `SI`, mientras que Gemini mantiene un comportamiento más equilibrado.

---

## Sesgo Común a Todos los Modelos

Todos los modelos tienden a sobre-predecir la clase `SI`. La causa probable es doble: la clase `SI` está ligeramente sobrerrepresentada en el dataset (37 vs 34 imágenes) y los criterios visuales para `NO` son más heterogéneos (hay varios tipos de posicionamiento incorrecto).

---

## Lecciones Aprendidas

1. **Feature extraction sin fine-tuning puede ser peor que el azar.** Congelar todo el backbone de ResNet18 dio un 26,67%.
2. **Data augmentation no es gratis.** Aumentar el recall de una clase a costa de balanced accuracy no siempre es una mejora.
3. **Los VLMs zero-shot son competitivos sin entrenamiento específico.** Gemini superó a todos los modelos entrenados con 71 imágenes.
4. **El tamaño del dataset es el cuello de botella real.** 71 imágenes son insuficientes para entrenar CNNs con confianza estadística.
5. **El error dominante en todos los modelos es el mismo.** Resolver el desbalance de clase `NO` mejoraría todos los modelos simultáneamente.

---

## Instalación

```bash
pip install -r requirements.txt
```

### Dependencias principales

| Librería | Uso |
|----------|-----|
| `torch`, `torchvision` | Entrenamiento de CNNs (ResNet18, MobileNetV2) |
| `scikit-learn` | Métricas: accuracy, F1, matriz de confusión |
| `Pillow` | Procesamiento de imágenes |
| `matplotlib`, `seaborn` | Visualización de resultados |
| `tqdm` | Barras de progreso durante el entrenamiento |
| `openai` | Cliente compatible para acceder a Gemini via OpenRouter |
| `python-dotenv` | Gestión segura de la API key |

### Configuración de la API (Grupo 3)

Crea un fichero `.env` en el directorio de los notebooks del Día 3:

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
```

> Obtén tu clave en [https://openrouter.ai/](https://openrouter.ai/) · **Nunca subas tu API key a GitHub.**

---

## Aviso Metodológico

Los conjuntos de test del Grupo 2 (15 imágenes) y del Grupo 3 (70 imágenes) no son directamente comparables. Un solo acierto o fallo en el Grupo 2 mueve la accuracy un 6,7%, frente al 1,4% en el Grupo 3. La comparativa es indicativa, no concluyente.

---

## Equipo

**Proyecto**: ProyectoFinal_CSEBDIA · Grupos 2, 3 y 5  
**Integrantes Grupo 5**: Carlos, Manel, Sergi, Fadoua  
**Fecha**: Mayo 2026
