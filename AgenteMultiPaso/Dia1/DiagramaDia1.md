# Día 1 — Diseño del Agente Multi-Paso

## Objetivo

Construir un agente que encadena tres modelos en secuencia:
1. **MedSAM** segmenta el dedo y aísla la región de interés
2. **ResidualCNN** clasifica SI/NO con una puntuación de confianza
3. **Gemini 2.0 Flash Lite** actúa como árbitro únicamente cuando la confianza de la CNN es baja

---

## Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ENTRADA: imagen PNG                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PASO 1 — segmentar()                                               │
│  MedSAM ViT-B                                                       │
│  · Bounding box automático (inversión → umbral=100 → contorno más  │
│    grande → +30px margen)                                           │
│  · predictor.predict(box, multimask_output=True)                    │
│  · Seleccionar máscara con score más alto                           │
│  · Salida: máscara binaria + score_medsam                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │  máscara + score_medsam
                             ▼
                    ┌────────────────┐
                    │ score ≥ 0.60?  │
                    └───┬────────┬───┘
                      NO│        │SÍ
                        │        ▼
                        │  usar máscara para
                        │  enmascarar fondo
                        └────────┬───────────
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PASO 2 — clasificar()                                              │
│  ResidualCNN (ResidualCNN.pth)                                 │
│  · Aplicar máscara a la imagen (fondo a negro)                      │
│  · Resize 224×224, normalizar [−1, 1]                              │
│  · Softmax → (prob_no, prob_si)                                     │
│  · Salida: etiqueta ("si"/"no") + confianza = max(probs)           │
└────────────────────────────┬────────────────────────────────────────┘
                             │  etiqueta + confianza
                             ▼
                  ┌──────────────────────┐
                  │ necesita_arbitraje() │
                  │ confianza < 0.70 ?   │
                  └───────┬──────┬───────┘
                       SÍ │      │ NO
                          │      │
                          ▼      ▼
┌───────────────────┐    ┌────────────────────────────────────────────┐
│  PASO 3 —         │    │  VEREDICTO FINAL = resultado CNN           │
│  consultar_       │    │  fuente = "cnn"                            │
│  gemini()         │    └────────────────────────────────────────────┘
│                   │
│  · imagen + overlay
│    verde (máscara)│
│  · prompt con     │
│    etiqueta CNN + │
│    confianza CNN  │
│  · Respuesta JSON │
│    {veredicto,    │
│     confianza,    │
│     justificacion}│
└────────┬──────────┘
         │
         ▼
┌──────────────────────┐
│ veredicto válido?    │
└────┬──────────┬───────┘
   SÍ│          │NO (error API)
     │          ▼
     │   VEREDICTO FINAL = CNN (fallback)
     │   fuente = "cnn_fallback"
     ▼
┌──────────────────────────────────────────┐
│  VEREDICTO FINAL = resultado Gemini      │
│  fuente = "gemini"                       │
└──────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SALIDA: veredicto_final()                                          │
│  {                                                                  │
│    veredicto_final: "si" / "no",                                    │
│    fuente: "cnn" | "gemini" | "cnn_fallback",                      │
│    confianza_final: float,                                          │
│    arbitraje_activado: bool,                                        │
│    resultado_segmentacion: {...},                                   │
│    resultado_cnn: {...},                                            │
│    resultado_gemini: {...} | None                                   │
│  }                                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Decisiones de Diseño

### Modelo CNN elegido: ResidualCNN

De los tres modelos disponibles, se usa `ResidualCNN.pth` porque:
- Es la arquitectura **recomendada** (⭐) en el notebook de entrenamiento
- Tiene conexiones residuales que estabilizan gradientes en datasets pequeños
- ~2.8M parámetros: equilibrio entre capacidad y sobreajuste con 70 imágenes

### Umbral de confianza: 0.70

Los scores de softmax del notebook de evaluación oscilan entre 0.51–0.62 para DeepCNN. La ResidualCNN puede producir scores más calibrados, pero el umbral de 0.70 garantiza que Gemini intervenga en los casos de genuina incertidumbre. Se puede ajustar tras los experimentos del Día 3.

**Consecuencia esperada**: con el dataset pequeño y las CNNs entrenadas con 48 imágenes, es probable que Gemini intervenga en >50% de los casos. Esto es informativo, no un fallo del diseño.

### Prompt de arbitraje de Gemini

Gemini recibe:
- La imagen con **overlay verde** (máscara de segmentación visible)
- La predicción de la CNN y su confianza (contexto explícito)
- Instrucción de responder en JSON estructurado

El overlay sirve para que Gemini vea exactamente qué región detectó MedSAM como dedo, lo que le da contexto espacial adicional al VLM.

### Gestión de fallos

Si Gemini falla (error de API, respuesta no parseable), el agente hace **fallback a la CNN** y documenta `fuente = "cnn_fallback"`. El agente nunca queda sin veredicto.

---

