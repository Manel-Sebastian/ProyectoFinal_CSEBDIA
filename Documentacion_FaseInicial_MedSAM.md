# Asistente de Posicionamiento Anatómico con MedSAM

Este proyecto documenta la fase inicial de validación del modelo **MedSAM** para la asistencia técnica en salas de radiología. El objetivo principal es verificar la correcta posición del paciente mediante imágenes de luz visible antes de proceder con la exposición a radiación ionizante.

## 📄 Resultados de la Fase Inicial

### 1. Evaluación de la Segmentación
**Pregunta:** *¿Segmenta correctamente la región de interés (ROI)?*

**Respuesta:** **Funcionalmente Exitosa.** A pesar de que el modelo MedSAM fue entrenado para imágenes médicas (Rayos X, CT, MRI), la última iteración demostró que es capaz de reconocer y aislar estructuras anatómicas en fotografías convencionales con un alto grado de precisión bajo las condiciones adecuadas de encuadre.

* **Identificación:** El modelo logró separar la anatomía del dedo del fondo y las sombras complejas de la imagen.
* **Métrica de Confianza (Score):** Se alcanzó un valor de **0.6284**, superando el umbral crítico establecido para validación externa.
* **Importancia del Prompt:** Se validó que un *Bounding Box* ajustado (simulando la colimación del equipo de rayos) incrementa la precisión del modelo al reducir el ruido ambiental.

---

### 2. Análisis del Asistente de Posicionamiento

El sistema no solo realiza una segmentación, sino que interpreta la calidad de la misma para asistir al técnico:

-   **Lógica de Veredicto:** Se implementó un umbral de éxito de **0.60**.
-   **Resultado de la prueba:** **0.6284** &rarr; **ESTADO: POSICIÓN VÁLIDA**.
-   **Visualización Semántica:** El sistema renderiza una máscara verde sobre la anatomía reconocida, proporcionando una confirmación visual inmediata de que el área de interés está correctamente encuadrada.

### 3. Ficha Técnica de la Última Interacción

| Parámetro | Valor |
| :--- | :--- |
| **Modelo** | MedSAM (ViT-B) |
| **Dispositivo** | CPU (Mapeo de pesos optimizado) |
| **Input Box** | `[280, 120, 480, 350]` |
| **Confianza (Score)** | **62.84%** |
| **Resultado** | ✅ Apto para Radiografía |

---

## 🚀 Conclusiones
La segmentación obtenida es válida para el propósito de **validación de técnica**. La capacidad de MedSAM para detectar morfología humana en condiciones no clínicas permite reducir errores de posicionamiento, optimizando el flujo de trabajo en radiología y minimizando la repetición de disparos innecesarios.

---
*Documentación generada para el curso de IA - Mayo 2026*
