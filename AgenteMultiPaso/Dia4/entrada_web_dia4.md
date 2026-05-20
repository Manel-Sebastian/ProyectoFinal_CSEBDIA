# MedSAM + Gemini + modelo propio: el pipeline completo

**Grupo 5 — Proyecto IA RX · Mayo 2026**

---

## El problema

Un escáner biométrico no funciona si el dedo no está bien colocado. Con solo **71 imágenes** de radiografías térmicas, el reto era construir un clasificador que distinga entre posicionamiento correcto (`SI`) e incorrecto (`NO`) — y que lo haga de forma fiable con tan pocos datos.

La solución no fue un único modelo. Fue una **cadena de tres IAs** donde cada modelo hace lo que mejor sabe hacer.

---

## La arquitectura: tres modelos en secuencia

```
Imagen PNG
    │
    ▼ MedSAM — segmenta el dedo
    │ si score < 0.60 → ERROR
    │
    ▼ ResidualCNN — clasifica SI/NO
    │ si p_cnn >= 0.70 → VEREDICTO DIRECTO
    │
    ▼ Gemini 2.0 Flash Lite — árbitro visual
      veredicto final + razón en texto
```

### Por qué esta secuencia

Cada modelo tiene un rol claro que justifica su posición:

**MedSAM** (Medical Segment Anything Model) no clasifica — segmenta. Su única función es delimitar el contorno del dedo en la imagen, generando una máscara binaria. Al enmascarar los píxeles de fondo antes de pasar la imagen a la CNN, la red no "ve" el escáner, la mesa, ni nada irrelevante — solo el dedo. Esto es especialmente importante con 71 imágenes: cualquier fuente de ruido visual penaliza mucho más en datasets pequeños.

**ResidualCNN** es el clasificador principal. Se entrenó sobre el dataset con fine-tuning parcial (capas `layer3`, `layer4` y `fc` descongeladas) y alcanzó un 60% de accuracy en test. El dato clave: cuando su probabilidad de salida (`softmax`) supera el 70%, el agente confía en ella directamente.

**Gemini 2.0 Flash Lite** entra solo cuando la CNN duda. Con un 70% de accuracy zero-shot (sin haber visto ninguna imagen del dominio), Gemini es un árbitro mejor que la CNN en los casos límite. Y al recibir también la máscara de MedSAM como segunda imagen, puede razonar sobre la región de interés específica.

---

## El código: la clase `AgenteMedico`

El pipeline completo está encapsulado en una clase con cinco métodos:

```python
class AgenteMedico:
    UMBRAL_CNN    = 0.70  # por debajo → Gemini arbitra
    UMBRAL_MEDSAM = 0.60  # por debajo → imagen inválida

    def segmentar(self, img_array):
        """MedSAM con bbox central. Lanza ValueError si score < 0.60."""

    def clasificar(self, img_array, mask):
        """CNN sobre imagen enmascarada. Devuelve (clase, p_cnn)."""

    def necesita_arbitraje(self, p_cnn):
        """True si p_cnn < 0.70."""

    def consultar_gemini(self, img_array, mask, pred_cnn, p_cnn):
        """Gemini recibe dos imágenes + contexto CNN. Devuelve (clase, razón)."""

    def veredicto_final(self, ruta_imagen):
        """Orquesta los 3 pasos. Devuelve dict con trazabilidad completa."""
```

El método `veredicto_final()` devuelve un diccionario que registra cada paso:

```python
{
    "imagen"          : "img_045.png",
    "score_medsam"    : 0.6357,
    "clase_cnn"       : "SI",
    "p_cnn"           : 0.8421,
    "arbitraje_gemini": False,
    "clase_final"     : "SI",
    "razon_gemini"    : None,
}
```

Esto permite auditar cualquier decisión sin reejecutar nada.

---

## El prompt de Gemini: la parte más difícil

Diseñar el prompt fue un proceso iterativo. La versión final que funcionó:

```
El escáner tiene una franja luminosa (zona brillante). El dedo entero
debe quedar centrado encima de esa franja, sin desplazarse ni salirse.
Ese es el único criterio.

CORRECTO (SI):
- El dedo está colocado encima de la franja luminosa y centrado en ella.
- El dedo no se sale de la franja ni por los lados ni por arriba/abajo.
- El dedo cubre la franja de forma plana, sin estar girado ni inclinado.

INCORRECTO (NO):
- El dedo está desplazado: parte del dedo queda fuera de la franja.
- El dedo solo toca un extremo o un lado de la franja, no el centro.
- El dedo está girado o inclinado de modo que no cubre bien la franja.

Responde ÚNICAMENTE con este formato:
VEREDICTO: [SI / NO]
RAZÓN: [Una sola frase con el criterio visual determinante]
```

**Lo que no funcionó**: mencionar "yema del dedo". Gemini se fijaba entonces solo en la punta en lugar del dedo completo, bajando el accuracy. La referencia a "la franja luminosa" es mucho más concreta y alineada con lo que el pipeline de Oscar detecta con `detect_light_window`.

**Por qué dos imágenes**: Gemini recibe la fotografía original y el overlay verde de MedSAM. La segunda imagen le indica explícitamente qué zona del escáner detectó el sistema — Gemini puede así confirmar o rebatir con criterio visual real.

---

## La integración con Oscar: bbox preciso para MedSAM

El pipeline de Oscar (`day1_opencv_helpers.py`) aporta la detección geométrica del escáner. En lugar de darle a MedSAM un bounding box genérico del 80% central de la imagen, se usa la intersección de la mano con la zona luminosa:

```
normalize → detect_black_panel → warp_panel → segment_hand
                                            → detect_light_window
                                            → hand_mask ∩ light_mask
                                            → proyección inversa (inv(warp_matrix))
                                            → bbox preciso del dedo
```

Además, el centro del bbox se pasa a MedSAM como punto de `foreground`, lo que reduce falsos positivos cuando el bbox todavía tiene algo de margen.

---

## Resultados

| Modelo | Accuracy test | Observación |
|--------|--------------|-------------|
| ResNet18 zero-shot | 53% | Prácticamente azar |
| ResNet18 feature extraction | 27% | Peor que el azar |
| **ResidualCNN fine-tuning** | **60%** | Mejor CNN del proyecto |
| Gemini zero-shot | **70%** | Sin entrenamiento |
| **Agente completo** | *ver Día 3* | MedSAM + CNN + Gemini |

El agente interviene con Gemini en los casos donde la CNN tiene `p_cnn < 0.70`. En el resto, decide directamente sin llamar a la API — lo que reduce latencia y coste.

### Cuándo Gemini ayuda y cuándo no

Del análisis del Día 3:

- **Gemini corrige a CNN**: casos donde la CNN predice incorrectamente con baja confianza — el razonamiento visual de Gemini sobre la franja luminosa detecta el error
- **Gemini empeora**: casos genuinamente ambiguos donde cualquier modelo falla — el posicionamiento real es borderline
- **Ambos coinciden correctamente**: casos claros que tanto la CNN como Gemini resuelven bien

El patrón dominante de error en todos los modelos es el mismo: sobre-predicción de `SI`. La causa probable es el desbalance de clases (37 SI vs 34 NO) combinado con la heterogeneidad de los tipos de `NO` (hay muchos tipos de posicionamiento incorrecto pero el correcto es siempre el mismo).

---

## Lecciones aprendidas

**Sobre el pipeline multi-modelo**: encadenar modelos especializados funciona mejor que un único modelo generalista cuando el dataset es pequeño. Cada modelo hace exactamente lo que le toca y no más.

**Sobre los VLMs como árbitros**: Gemini con 70% zero-shot supera a la CNN entrenada con 60%. Para datasets pequeños, la capacidad de razonamiento visual de un VLM puede compensar la falta de datos de entrenamiento.

**Sobre el prompting**: el criterio visual tiene que ser tan concreto como sea posible. "El dedo centrado en la franja luminosa" es mucho más preciso para Gemini que descripciones genéricas de posicionamiento.

**Sobre el umbral**: el 0.70 es un equilibrio entre calidad (Gemini arbitra cuando hace falta) y coste (no se llama a la API para casos fáciles). El análisis de sensibilidad del Día 3 muestra qué ocurre con otros valores.

---

## Código y notebooks

| Fichero | Descripción |
|---------|-------------|
| `AgenteMultiPaso/Dia2/agente_medico_dia2.ipynb` | Implementación completa del agente |
| `AgenteMultiPaso/Dia3/agente_medico_dia3.ipynb` | Evaluación sobre el set de test |
| `AgenteMultiPaso/Dia4/demo_agente_dia4.ipynb` | Demostración visual paso a paso |

---

*Grupo 5 — Carlos, Manel, Sergi, Fadoua · Proyecto IA RX · Mayo 2026*
