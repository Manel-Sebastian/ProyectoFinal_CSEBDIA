# Día 4 — Documentación + Publicación Web (5h)

**Grupo 5 — Agente multi-paso (MedSAM + CNN + Gemini)**  
**Proyecto**: Clasificación de posicionamiento de dedo en escáner biométrico  
**Fecha**: Mayo 2026

---

## Objetivo del día

Documentar el agente completo, preparar un notebook de demostración presentable y redactar la entrada web del proyecto.

| Tarea | Descripción | Tiempo |
|-------|-------------|--------|
| Tarea 1 | Documentar el agente: diagrama de flujo, código, análisis de cuándo y cómo interviene Gemini | 1.5h |
| Tarea 2 | Notebook de demostración con ejemplos del flujo completo paso a paso | 1.5h |
| Tarea 3 | Entrada web: "MedSAM + Gemini + modelo propio: el pipeline completo" | 2h |

---

## Tarea 1 — Documentación del agente *(1.5h)*

La documentación del agente se consolidó en los readmes de cada día:

- **`Dia1/readme.md`**: diseño del flujo, prompt de Gemini, umbral de confianza, diagrama Mermaid
- **`Dia2/readme.md`**: implementación detallada de los 5 métodos de `AgenteMedico`, integración Oscar+MedSAM, decisiones de implementación
- **`Dia3/readme.md`**: evaluación completa, comparativa CNN vs agente, análisis de intervenciones de Gemini, sensibilidad del umbral

### Diagrama de flujo del agente

```
╔══════════════════════════════════════════════════════════════════╗
║           PIPELINE AGENTE MEDICO — GRUPO 5                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   [Imagen PNG 640×480]                                          ║
║          │                                                       ║
║          ▼  PASO 1 — MedSAM (ViT-B, ~90M params)               ║
║   score >= 0.60?  No → ERROR: imagen no válida                  ║
║          │ Sí                                                    ║
║          ▼  PASO 2 — ResidualCNN (2.79M params)                 ║
║   p_cnn >= 0.70?  Sí → VEREDICTO CNN (alta confianza)          ║
║          │ No                                                    ║
║          ▼  PASO 3 — Gemini 2.0 Flash Lite (zero-shot)          ║
║   VEREDICTO GEMINI + razón en texto                             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

### Cuándo y cómo interviene Gemini

Gemini **no interviene siempre** — solo cuando `p_cnn < 0.70`. El análisis del Día 3 muestra cuatro tipos de intervención:

| Tipo | CNN | Gemini | Efecto |
|------|-----|--------|--------|
| Gemini corrige | ✗ falla | ✓ acierta | El agente mejora respecto a la CNN sola |
| Gemini empeora | ✓ acierta | ✗ falla | El agente empeora respecto a la CNN sola |
| Ambos correctos | ✓ | ✓ | Gemini confirma el acierto |
| Ambos incorrectos | ✗ | ✗ | El caso es difícil para ambos modelos |

El patrón de las razones de Gemini permite entender qué criterios visuales usa y si el prompt está bien calibrado.

---

## Tarea 2 — Notebook de demostración *(1.5h)*

`demo_agente_dia4.ipynb` es el notebook presentable del proyecto. A diferencia de los notebooks de Día 2 y 3 (orientados a implementación y evaluación), este notebook es una **demostración visual paso a paso** diseñada para que cualquier persona entienda el pipeline.

### Estructura del notebook

**Selección automática de imágenes**: el notebook busca automáticamente la imagen con mayor `p_cnn` (Demo 1) y la de menor `p_cnn` (Demo 2) para mostrar los dos caminos posibles del agente.

**Demo 1 — CNN decide directamente** (5 columnas):

| ① | ② | ③ | ④ | ⑤ |
|---|---|---|---|---|
| Imagen original | Máscara MedSAM overlay | CNN input (fondo negro) | Barras de probabilidad CNN | Veredicto final (azul) |

La barra de probabilidad muestra claramente que la CNN supera el umbral 0.70 — el agente no necesita llamar a Gemini.

**Demo 2 — Gemini arbitra** (dos visualizaciones separadas):

*Pasos 1+2*: igual que Demo 1 pero la barra de CNN queda por debajo del umbral — se ve visualmente que la CNN no es segura.

*Paso 3*: Gemini recibe dos imágenes (original + overlay verde) y emite su veredicto con una razón en texto. Se muestra qué envió el agente y qué respondió Gemini.

**Cuadro comparativo**: las dos demos en el mismo grid 2×5 para comparar side-by-side los dos caminos del pipeline.

**Diagrama de flujo en texto**: representación ASCII del pipeline completo, reproducible sin dependencias externas.

---

## Tarea 3 — Entrada web *(2h)*

`entrada_web_dia4.md` contiene la entrada web completa, lista para publicar.

**Título**: *MedSAM + Gemini + modelo propio: el pipeline completo*

**Estructura**:
1. El problema (71 imágenes, dataset pequeño, reto de clasificación)
2. La arquitectura: por qué tres modelos en secuencia y no uno solo
3. El código: la clase `AgenteMedico` y sus 5 métodos
4. El prompt de Gemini: proceso iterativo y por qué la versión final funciona
5. La integración con Oscar: bbox preciso mediante intersección mano∩luz
6. Resultados: tabla comparativa de todos los modelos del proyecto
7. Cuándo Gemini ayuda y cuándo no (análisis cuantitativo del Día 3)
8. Lecciones aprendidas

**Puntos clave de la entrada**:
- Explica el razonamiento detrás de cada decisión de diseño (no solo el qué sino el por qué)
- Documenta las iteraciones del prompt (qué falló con "yema" y por qué funciona "franja luminosa")
- Contextualiza los resultados dentro de los resultados del proyecto completo (Grupos 2, 3 y 5)

---

## Ficheros del Día 4

| Fichero | Descripción |
|---------|-------------|
| `demo_agente_dia4.ipynb` | Notebook de demostración visual (22 celdas) |
| `entrada_web_dia4.md` | Entrada web publicable |
| `readme.md` | Este fichero — documentación del Día 4 |

---

## Resumen del Día 4

| Tarea | Tiempo | Resultado |
|-------|--------|-----------|
| Documentación del agente | 1.5h | ✅ Diagrama de flujo, análisis de intervenciones, readmes consolidados |
| Notebook de demostración | 1.5h | ✅ 22 celdas, dos demos visuales, cuadro comparativo |
| Entrada web | 2h | ✅ Documento completo listo para publicar |

**Total**: 5h

---

*Documentación del Día 4 — Grupo 5 — Proyecto IA RX — Mayo 2026*
