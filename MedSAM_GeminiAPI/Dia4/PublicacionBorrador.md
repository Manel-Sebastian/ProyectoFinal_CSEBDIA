# Cómo encadenamos tres IAs para analizar radiografías

**Proyecto Final CESBDIA** · Clasificación biométrica de posicionamiento de dedo · Grupos 2 y 3

---

Un experimento académico que arrancó con una CNN clásica, pasó por segmentación médica especializada y terminó comparando resultados contra un modelo de lenguaje multimodal de última generación. Esto es lo que aprendimos.

---

## El problema que queríamos resolver

Un escáner biométrico no funciona bien si el dedo no está colocado correctamente. El objetivo era construir un sistema capaz de clasificar automáticamente si la posición del dedo en el escáner es válida (`si`) o no (`no`). Parece sencillo. Con 71 imágenes de dataset, resultó ser cualquier cosa menos trivial.

El equipo se dividió en dos grupos con enfoques distintos. El resultado fue una comparativa que pone en tensión los métodos clásicos de visión por computador contra los grandes modelos multimodales.

---

## Las tres IAs de la cadena

No fue un pipeline lineal diseñado desde el principio, sino una arquitectura que emergió de la experimentación. Cada IA cubría una capa distinta del problema.

| Paso | Modelo | Rol |
|------|--------|-----|
| 1 | **ResNet18 / MobileNetV2** | CNNs preentrenadas en ImageNet. Backbone de extracción de características visuales. Fine-tuning sobre el dominio del escáner. |
| 2 | **MedSAM** | Segment Anything Model adaptado para imagen médica. Segmentación del dedo dentro del encuadre del sensor para aislar la región de interés. |
| 3 | **Gemini 2.0 Flash Lite** | Vision-Language Model vía OpenRouter. Zero-shot multimodal: recibe la imagen y un prompt estructurado, devuelve JSON con clasificación y razonamiento. |

> **MedSAM no clasifica —prepara el terreno.** Su tarea es aislar el dedo del ruido de fondo para que el clasificador final tenga algo limpio sobre lo que razonar.

La idea detrás de la cadena: usar MedSAM para limpiar la imagen antes de la clasificación. Un segmentador especializado en imagen médica es mucho más fiable para delimitar estructuras anatómicas que dejar esa tarea implícita dentro de una CNN genérica.

---

## Grupo 2: el camino clásico (CNNs + fine-tuning)

### ResNet18 — tres estrategias, tres resultados muy distintos

El Grupo 2 exploró ResNet18 con tres aproximaciones progresivas sobre el mismo backbone.

**Zero-Shot (sin entrenar):** Accuracy del 53,33%. Básicamente moneda al aire: los pesos de ImageNet no tienen ninguna alineación con imágenes de escáner biométrico y la cabeza de clasificación es aleatoria.

**Feature Extraction (backbone congelado):** Aquí vino la sorpresa negativa. Solo el 26,67% de accuracy —peor que el azar. Las features que ResNet18 aprendió con ImageNet no son transferibles a este dominio sin adaptar al menos las capas profundas. Congelar todo el backbone fue un error.

**Fine-Tuning (layer3 + layer4 + fc descongelados):** El único enfoque que funciona, con un 60% de accuracy. El modelo consigue un recall del 100% en la clase `si`, aunque a costa de muchos falsos positivos (precision del 57%). En la práctica, está prediciendo casi todo como `si`.

### MobileNetV2 — el efecto (nulo) del data augmentation

Con MobileNetV2 el equipo probó si enriquecer los datos con transformaciones artificiales (pipeline de Albumentations) mejoraría los resultados.

La respuesta: no en términos de accuracy global. Ambas variantes se quedaron en el 53,33%. Lo que sí cambió fue el perfil del modelo: la versión con augmentation aumentó el recall de `si` del 62,5% al 87,5%, pero redujo la balanced accuracy. Un modelo más sensible pero menos equilibrado.

> **Nota sobre VGG16:** aparece referenciado en el notebook como comparativa arquitectónica teórica, pero no llegó a entrenarse. Se descartó por su coste computacional frente a MobileNetV2.

---

## Grupo 3: el atajo multimodal (Gemini zero-shot)

El Grupo 3 tomó una dirección completamente diferente: en lugar de entrenar modelos, utilizó Gemini 2.0 Flash Lite a través de la API de OpenRouter con un prompt zero-shot.

El prompt incluía descripciones explícitas de ambas clases y solicitaba una respuesta en formato JSON estructurado. Sin una sola imagen de entrenamiento específica del dominio.

### Resultados

| Métrica | Valor |
|---------|-------|
| Accuracy global | **70,00%** |
| Aciertos | 49 / 70 |
| Macro Precision | 70,35% |
| Macro Recall | 69,77% |
| Macro F1 | 69,70% |

**Por clase:**

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|----|
| `SI` | 68,29% | 77,78% | 72,73% |
| `NO` | 72,41% | 61,76% | 66,67% |

**Matriz de confusión:**

|  | Pred. SI | Pred. NO |
|--|----------|----------|
| **Real SI** | 28 (TP) | 8 (FN) |
| **Real NO** | 13 (FP) | 21 (TN) |

Gemini detecta bien las posturas correctas (recall SI = 77,78%) pero falla más en identificar las incorrectas (recall NO = 61,76%). Los falsos positivos —13 imágenes `NO` clasificadas como `SI`— son el error dominante.

---

## Comparación

### Gemini vs. mejor modelo del Grupo 2 (ResNet18 Fine-Tuning)

| Métrica | Gemini (G3) | ResNet18 FT (G2) | Diferencia |
|---------|-------------|------------------|------------|
| Accuracy | **70,00%** | 60,00% | +10 pp Gemini |
| F1 (si) | **72,73%** | 72,73% | Empate |
| Precision (si) | 68,29% | 57,14% | +11,15 pp Gemini |
| Recall (si) | 77,78% | **100,00%** | +22,22 pp ResNet18 |

El F1 de la clase `si` es idéntico (72,73%) pero conseguido de formas opuestas:

- **ResNet18 FT** maximiza recall prediciendo casi todo como `si`. Recall perfecto, precision mediocre.
- **Gemini** tiene un comportamiento más equilibrado. Precision y recall son más cercanos entre sí.

Gemini gana en accuracy global y en macro F1. Es un modelo menos sesgado hacia una clase.

### Comparando zero-shots: Gemini vs. ResNet18 sin entrenar

| Métrica | Gemini | ResNet18 Zero-Shot |
|---------|--------|--------------------|
| Accuracy | **70,00%** | 53,33% |
| F1 (si) | **72,73%** | 53,33% |

La diferencia (+16,67 pp) se explica por la naturaleza de cada modelo. ResNet18 zero-shot tiene una cabeza de clasificación aleatoria y ningún alineamiento con la tarea. Gemini es un VLM entrenado explícitamente para razonar sobre imágenes siguiendo instrucciones textuales.

---

## El sesgo que todos comparten

Hay un patrón que atraviesa todos los modelos:

| Modelo | Sesgo dominante |
|--------|-----------------|
| ResNet18 FT | Sobre-predice `si` (recall 100%, precision 57%) |
| MobileNetV2 + Aug | Sobre-predice `si` (recall 87,5%, precision 53,85%) |
| Gemini | Sobre-predice `si` (13 FP vs 8 FN), pero de forma más moderada |

La causa probable es doble: la clase `si` está ligeramente sobrerrepresentada en el dataset de entrenamiento (37 vs 34 imágenes) y los criterios visuales para la clase `no` son más heterogéneos. Hay varios tipos de posicionamiento incorrecto, lo que hace que esa clase sea más difícil de aprender.

---

## Aviso metodológico importante

> Los conjuntos de test no son comparables directamente. El Grupo 2 evaluó sobre **15 imágenes**; el Grupo 3 sobre **70**. Un solo acierto o fallo en G2 mueve la accuracy un 6,7%, mientras que en G3 solo un 1,4%. Las métricas del Grupo 2 tienen una varianza estadística mucho mayor. La comparación es indicativa, no concluyente.

Para una comparación rigurosa, ambos grupos deberían evaluarse sobre el mismo conjunto de test.

---

## Lecciones aprendidas

**1. Feature extraction sin fine-tuning puede ser peor que el azar.**  
Congelar todo el backbone de ResNet18 y entrenar solo la cabeza lineal dio un 26,67%. Las features de ImageNet no son transferibles a dominio de escáner sin al menos descongelar las capas profundas.

**2. Data augmentation no es gratis.**  
Aumentar el recall de una clase a costa de balanced accuracy no siempre es una mejora. Depende de qué error es más costoso en producción.

**3. Los VLMs zero-shot son competitivos sin entrenamiento específico.**  
Gemini superó a todos los modelos entrenados con 71 imágenes. En datasets pequeños, un modelo grande con buen prompting puede ser más efectivo que entrenar una CNN desde cero.

**4. El tamaño del dataset es el cuello de botella real.**  
71 imágenes son insuficientes para entrenar modelos CNN con confianza estadística. Antes de ajustar hiperparámetros, lo más impactante sería ampliar el dataset.

**5. El error dominante en todos los modelos es el mismo.**  
La tendencia a sobre-predecir `si` es un síntoma del desbalance y la heterogeneidad de la clase `no`. Resolver esto —ya sea con más datos de clase `no` o con pesos de clase en la función de pérdida— mejoraría todos los modelos simultáneamente.

---

