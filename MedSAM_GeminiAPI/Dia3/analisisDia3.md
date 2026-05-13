# Clasificación Zero-Shot con Gemini — Escáner Biométrico de Dedo

Repositorio con dos notebooks de Jupyter que evalúan modelos multimodales de Google (Gemini 2.0 Flash Lite) en una tarea de **clasificación binaria zero-shot** sobre imágenes de un escáner biométrico de huella dactilar. El objetivo es determinar si un dedo está correctamente posicionado sobre la barra de luz del escáner (`SI`) o no (`NO`), sin entrenamiento previo del modelo en este dominio.

---

## Notebooks

### 1. `ZeroShot_GeminiAPI.ipynb`

Evaluación directa de Gemini 2.0 Flash Lite sobre un conjunto de 70 imágenes del escáner biométrico, accedido a través de **OpenRouter** con el cliente compatible con la API de OpenAI.

**Pipeline:**
1. Instalación de dependencias y configuración del cliente OpenRouter
2. Verificación de conexión con el modelo
3. Preprocesamiento de imágenes (conversión a base64, normalización RGB)
4. Carga del dataset desde `imagenes_procesadas/train/{si,no}` y `test/{si,no}`
5. Diseño del prompt zero-shot con descripción detallada de ambas clases
6. Prueba de inferencia multimodal individual
7. Muestreo estratificado: 36 imágenes `SI` + 34 imágenes `NO` (70 total)
8. Inferencia en lote con guardado incremental en `resultados_zero_shot.jsonl`
9. Cálculo de métricas y matriz de confusión

**Resultado:** Accuracy = **70.00%** sobre 70 imágenes.

**Salida del modelo (JSON):**
```json
{
  "veredicto": "SI" | "NO",
  "confianza": 0.0 – 1.0,
  "justificacion": "descripción breve"
}
```

---

### 2. `dia3_Gemini_Sergi.ipynb`

Notebook del **Día 3** del proyecto grupal (Grupo 5: Carlos, Manel, Sergi, Fadoua). Extiende la evaluación zero-shot añadiendo **tres tipos de prompt distintos** y una comparativa más exhaustiva, usando las mismas 70 imágenes del Día 2.

**Pipeline:**
1. Configuración del cliente OpenRouter y verificación de conexión
2. Carga de etiquetas reales desde `resultados_dia2.json`
3. Definición de tres prompts zero-shot:

| Prompt | Pregunta | Respuestas posibles |
|---|---|---|
| `posicion` | ¿Está el dedo bien posicionado? | `SI` / `NO` |
| `tipo_parte` | ¿Qué parte del cuerpo se ve? | `MANO` / `PIE` / `TORAX` / `OTRO` |
| `calidad` | ¿Qué calidad técnica tiene la imagen? | `BUENA` / `REGULAR` / `MALA` |

4. Función de clasificación con reintentos automáticos y manejo de rate limits
5. Clasificación en lote de las 70 imágenes con los 3 prompts
6. Cálculo de accuracy, precision, recall y F1
7. Visualizaciones: matriz de confusión, distribución real vs predicha, calidad por clase
8. Análisis de falsos positivos y falsos negativos con visualización de imágenes
9. Guardado de resultados en `resultados_openrouter_dia3.json` para uso en el Día 4

**Archivos generados:**
- `resultados_openrouter_dia3.json`
- `metricas_openrouter_dia3.png`
- `falsos_positivos_openrouter_dice_si_pero_era_no.png`
- `falsos_negativos_openrouter_dice_no_pero_era_si.png`

---

## Requisitos

```bash
pip install openai pillow scikit-learn matplotlib seaborn python-dotenv requests
```

## Configuración

Crea un fichero `.env` en el mismo directorio que los notebooks:

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
```

> Obtén tu clave gratuita en [https://openrouter.ai/](https://openrouter.ai/)

**⚠️ Nunca subas tu API key a GitHub.**

---

## Estructura de datos esperada

```
imagenes_procesadas/
├── train/
│   ├── si/   ← dedo correctamente posicionado
│   └── no/   ← dedo mal posicionado
└── test/
    ├── si/
    └── no/
```

---

## Modelo utilizado

**Gemini 2.0 Flash Lite** (`google/gemini-2.0-flash-lite-001`) accedido vía OpenRouter con el endpoint compatible con la API de OpenAI (`https://openrouter.ai/api/v1`).

---

## Análisis de errores

| Tipo de error | Causa probable |
|---|---|
| **Falso Positivo** (NO→SI) | El modelo sobreestima; manos giradas ≤30° las clasifica como correctas |
| **Falso Negativo** (SI→NO) | El modelo es estricto con el centrado; descarta imágenes válidas con poco margen |
| **Respuesta INVALIDO** | El modelo añade texto extra pese a la instrucción de responder solo en JSON |
| **Tipo_parte incorrecto** | Imágenes con bajo contraste o zoom extremo confunden al modelo |

---

## Contexto del proyecto

Estos notebooks forman parte de un proyecto multi-día de comparativa entre enfoques de visión por computador:

- **Día 2:** Segmentación con MedSAM y generación de etiquetas
- **Día 3:** Clasificación zero-shot con LLMs multimodales (este repositorio)
- **Día 4:** Comparativa final contra CNNs entrenadas (Grupos 2 y 3)
