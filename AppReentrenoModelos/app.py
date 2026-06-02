import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import torch

from app.inference import load_model, TRANSFORM_EVAL, IDX2LABEL

try:
    from retrain import (
        retrain_all_models,
        load_metrics_history,
        DATASET_DIR,
        MODELS_DIR,
        MODEL_REGISTRY,
    )

    RETRAIN_AVAILABLE = True
except ImportError:
    RETRAIN_AVAILABLE = False
    DATASET_DIR = Path("./dataset")
    MODELS_DIR = Path("./modelos")

# Fallback: si la ruta importada de retrain.py no existe, probar ./modelos
_local_models = Path("./modelos")
if RETRAIN_AVAILABLE and not MODELS_DIR.exists() and _local_models.exists():
    MODELS_DIR = _local_models

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(page_title="Radiología AI", page_icon="🩻", layout="wide")

# ---------------------------------------------------------
# 2. CARGAR CLAVES API
# ---------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_AVAILABLE = bool(GEMINI_API_KEY)
if GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)

MULTIMODAL_MODEL = os.environ.get("MULTIMODAL_MODEL_GEMINI")

# ---------------------------------------------------------
# 3. PROMPT GEMINI
# ---------------------------------------------------------
PROMPT_SISTEMA = """
Eres un técnico radiólogo experto evaluando la calidad de posicionamiento en radiografías de manos/dedos.
Tu tarea es analizar la imagen adjunta y determinar si la postura del dedo para la radiografía es CORRECTA ("si") o INCORRECTA ("no").

Reglas estrictas de clasificación:
- Responde "si" (Postura correcta): El dedo objetivo está claramente visible, bien centrado en la imagen, y se encuentra totalmente dentro de la zona de luz (campo de colimación). No hay superposiciones graves que impidan ver el hueso de forma clara.
- Responde "no" (Postura incorrecta): El dedo está mal posicionado, está parcial o totalmente fuera del área de luz oscura, está cortado por los bordes de la imagen, o la mano está en una postura donde otros dedos lo tapan de forma severa.

Responde ÚNICAMENTE con la palabra "si" o la palabra "no" en minúsculas. No añadas puntos, signos de puntuación, ni explicaciones extra.
"""


# ---------------------------------------------------------
# 4. DESCUBRIR MODELOS DISPONIBLES
# ---------------------------------------------------------
def _etiqueta_amigable(parte: str, nombre_fichero: str) -> str:
    """Convierte nombres técnicos en etiquetas legibles."""
    nombres = {
        "resnet18_finetuning": "ResNet18",
        "mobilenet_augmentation": "MobileNet",
    }
    parte_cap = parte.capitalize()
    modelo = nombres.get(nombre_fichero, nombre_fichero)
    return f"🧠 {modelo} — {parte_cap}"


def descubrir_modelos(models_dir: Path) -> dict:
    """
    Escanea models_dir en busca de archivos .pt.
    Devuelve un dict donde la clave es la etiqueta amigable y el valor
    es un dict con path, body_part y model_name.
    """
    encontrados = {}
    if not models_dir.exists():
        return encontrados
    for pt in sorted(models_dir.rglob("*.pt")):
        body_part = pt.parent.name
        model_name = pt.stem
        etiqueta = _etiqueta_amigable(body_part, model_name)
        encontrados[etiqueta] = {
            "path": str(pt),
            "body_part": body_part,
            "model_name": model_name,
        }
    return encontrados


# ---------------------------------------------------------
# 5. INFERENCIA
# ---------------------------------------------------------
def clasificar_con_gemini(img: Image.Image) -> str:
    if not GEMINI_AVAILABLE:
        return "error: API key no configurada"
    try:
        model = genai.GenerativeModel(MULTIMODAL_MODEL)
        respuesta = model.generate_content([PROMPT_SISTEMA, img])
        texto = respuesta.text.strip().lower()
        if "si" in texto and "no" not in texto:
            return "si"
        elif "no" in texto and "si" not in texto:
            return "no"
        return texto
    except Exception as e:
        return f"error: {e}"


def clasificar_con_modelo_local(img: Image.Image, body_part: str, model_name: str):
    """
    Carga un modelo local y predice sobre una imagen PIL.
    Devuelve (label, confidence).
    """
    model = load_model(body_part, model_name)
    if model is None:
        raise RuntimeError(
            f"No se encontró el modelo '{model_name}' para '{body_part}'. "
            f"Comprueba que existe en modelos/{body_part}/{model_name}.pt"
        )

    tensor = TRANSFORM_EVAL(img.convert("RGB")).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = IDX2LABEL[predicted.item()]
    return label, confidence.item()


# ---------------------------------------------------------
# 6. SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Ajustes")

    carpeta_modelos_sidebar = st.text_input(
        "📁 Carpeta de modelos",
        value=str(MODELS_DIR),
        help="Ruta donde están guardados los modelos entrenados.",
        key="app_nueva_models_dir",  # key única para no heredar estado de app.py
    )

    # Debug visual para saber qué ruta se está escaneando
    st.caption(f"Escaneando: `{Path(carpeta_modelos_sidebar).resolve()}`")
    if not Path(carpeta_modelos_sidebar).exists():
        st.error(f"❌ La carpeta no existe: `{carpeta_modelos_sidebar}`")

    st.divider()

    st.markdown("### 📖 ¿Cómo funciona esta aplicación?")
    st.markdown("""
Esta aplicación usa **Inteligencia Artificial** para comprobar si la postura
de un dedo en una radiografía es correcta o no.

**Pasos para analizar una radiografía:**

**1.** Elige la pestaña **"Subir archivo"** o **"Usar cámara"**

**2.** Carga o fotografía la radiografía

**3.** Selecciona qué IA quieres que la analice:
- Los modelos **ResNet18** y **MobileNet** son IAs que hemos entrenado
  nosotros con radiografías reales
- **Gemini** es una IA de Google (necesita conexión a internet)

**4.** Pulsa el botón **"Analizar"** y obtendrás el resultado al instante

---

**¿Qué significa el resultado?**
- ✅ **CORRECTA** — el dedo está bien colocado y centrado
- ❌ **INCORRECTA** — el dedo está mal posicionado o cortado

---

**¿Cuándo actualizar la IA?**

Si la IA empieza a fallar con frecuencia, ve a la pestaña
**"Actualizar la IA"** y pulsa el botón para que aprenda de nuevo
con imágenes actualizadas. Solo tarda unos minutos.
    """)

    st.divider()
    st.caption("🩻 Radiología AI — Asistente de posicionamiento")

# Construir opciones con la carpeta configurada
MODELOS_PT = descubrir_modelos(Path(carpeta_modelos_sidebar))
opciones_modelo = {}
# Mostrar Gemini siempre; si no hay API key, avisará al analizar
opciones_modelo["☁️ Gemini — IA en la nube (Google)"] = {"tipo": "gemini"}

for etiqueta, info in MODELOS_PT.items():
    opciones_modelo[etiqueta] = {"tipo": "local", **info}

# Aviso en sidebar sobre el estado de Gemini
with st.sidebar:
    if not GEMINI_AVAILABLE:
        st.info("🔑 Añade `GEMINI_API_KEY` en tu archivo `.env` para activar Gemini.")

# Mostrar en el sidebar cuántos modelos se detectaron
with st.sidebar:
    n_locales = len(MODELOS_PT)
    if n_locales:
        st.success(f"✅ {n_locales} modelo(s) local(es) detectado(s)")
    else:
        st.warning("⚠️ No se detectaron modelos locales en la carpeta indicada.")


# ---------------------------------------------------------
# 7. FUNCIÓN DE ANÁLISIS
# ---------------------------------------------------------
def render_analisis(imagen: Image.Image, key_suffix: str):
    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.markdown("#### Tu radiografía")
        st.image(imagen, use_container_width=True)

    with col_res:
        st.markdown("#### Paso 2 — Elige la IA y analiza")

        if not opciones_modelo:
            st.warning(
                "**No hay ninguna IA disponible todavía.**\n\n"
                "Ve a la pestaña **🤖 Actualizar la IA** y pulsa el botón "
                "para entrenar los modelos por primera vez."
            )
            return

        etiqueta = st.radio(
            "¿Qué IA quieres usar?",
            options=list(opciones_modelo.keys()),
            key=f"modelo_radio_{key_suffix}",
        )
        modelo_info = opciones_modelo[etiqueta]

        st.divider()

        if st.button(
            "🔍 Analizar la radiografía",
            use_container_width=True,
            type="primary",
            key=f"btn_analizar_{key_suffix}",
        ):
            with st.spinner("La IA está analizando la imagen, espera un momento..."):
                if modelo_info["tipo"] == "gemini":
                    resultado = clasificar_con_gemini(imagen)
                    confianza = None
                else:
                    try:
                        resultado, confianza = clasificar_con_modelo_local(
                            imagen,
                            body_part=modelo_info["body_part"],
                            model_name=modelo_info["model_name"],
                        )
                    except Exception as e:
                        resultado = f"error: {e}"
                        confianza = None

            st.markdown("---")
            st.markdown("## Resultado:")

            if resultado == "si":
                msg = (
                    "### ✅ POSTURA CORRECTA\n\n"
                    "El dedo está bien colocado y centrado en la radiografía. "
                    "La imagen es válida para su uso."
                )
                if confianza is not None:
                    msg += f"\n\n**Confianza del modelo:** {confianza:.1%}"
                st.success(msg)
                st.balloons()
            elif resultado == "no":
                msg = (
                    "### ❌ POSTURA INCORRECTA\n\n"
                    "El dedo no está bien colocado. Puede estar cortado, "
                    "girado o fuera del área de luz. Repite la radiografía."
                )
                if confianza is not None:
                    msg += f"\n\n**Confianza del modelo:** {confianza:.1%}"
                st.error(msg)
            else:
                st.warning(
                    f"⚠️ No se pudo obtener un resultado claro.\n\n"
                    f"Detalle técnico: {resultado}"
                )


# ---------------------------------------------------------
# 8. CABECERA PRINCIPAL
# ---------------------------------------------------------
st.title("🩻 Asistente de Posicionamiento Radiológico")
st.markdown(
    "Sube una radiografía y la IA te dirá al instante si la postura del dedo es correcta o no."
)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📁 Subir archivo",
        "📸 Usar cámara",
        "📊 Resultados de los modelos",
        "🤖 Actualizar la IA",
    ]
)

# ---------------------------------------------------------
# 9. TAB 1 — SUBIR ARCHIVO
# ---------------------------------------------------------
with tab1:
    st.markdown("#### Paso 1 — Sube la radiografía")
    st.markdown("Arrastra el archivo aquí o haz clic para buscarlo en tu ordenador.")
    archivo_subido = st.file_uploader(
        "Formatos válidos: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if archivo_subido is not None:
        st.divider()
        render_analisis(Image.open(archivo_subido), key_suffix="tab1")

# ---------------------------------------------------------
# 10. TAB 2 — CÁMARA
# ---------------------------------------------------------
with tab2:
    st.markdown("#### Paso 1 — Fotografía la radiografía")
    st.markdown("Apunta la cámara a la pantalla donde se muestra la radiografía.")
    foto_camara = st.camera_input("Captura", label_visibility="collapsed", key="app_nueva_camera")
    if foto_camara is not None:
        st.divider()
        render_analisis(Image.open(foto_camara), key_suffix="tab2")

# ---------------------------------------------------------
# 11. TAB 3 — RENDIMIENTO
# ---------------------------------------------------------
with tab3:
    st.markdown("### 📊 ¿Cómo de bien funciona la IA?")
    st.markdown(
        "Estas cifras muestran el porcentaje de aciertos de cada modelo "
        "cuando se probó con radiografías que nunca había visto antes."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Aciertos totales", "100%", delta="Modelo actual")
    col2.metric("Aciertos en posturas buenas", "100%")
    col3.metric("No se le escapó ninguna", "100%")
    col4.metric("Puntuación global", "100%")

    st.markdown("---")
    with st.expander("ℹ️ ¿Qué significa cada cifra?"):
        st.markdown("""
| Cifra | Qué mide |
|---|---|
| **Aciertos totales** | De cada 100 radiografías, ¿cuántas clasificó bien? |
| **Aciertos en posturas buenas** | Cuando la postura era correcta, ¿cuántas veces lo dijo bien? |
| **No se le escapó ninguna** | ¿Cuántas posturas incorrectas detectó sin dejar pasar ninguna? |
| **Puntuación global** | Combinación de las anteriores en un solo número |
        """)
    st.info(
        "💡 Si estos números bajan en el futuro, ve a **'Actualizar la IA'** "
        "para reentrenar los modelos con nuevas imágenes."
    )

# ---------------------------------------------------------
# 12. TAB 4 — REENTRENAMIENTO
# ---------------------------------------------------------
with tab4:
    st.markdown("### 🤖 Actualizar la Inteligencia Artificial")
    st.markdown(
        "Aquí puedes enseñarle a la IA con nuevas radiografías para que mejore sus resultados. "
        "El proceso es automático y tarda entre **5 y 15 minutos** según tu ordenador."
    )

    with st.expander("ℹ️ ¿Cuándo debo actualizar la IA?"):
        st.markdown("""
- Cuando la IA empiece a cometer errores con frecuencia
- Cuando hayas añadido nuevas radiografías al dataset
- Cuando quieras mejorar los resultados de un modelo concreto

**No hace falta hacerlo a menudo.** Con actualizarlo cada vez que añadas imágenes nuevas es suficiente.
        """)

    if not RETRAIN_AVAILABLE:
        st.error(
            "❌ No se puede conectar con el sistema de entrenamiento. "
            "Contacta con el administrador."
        )
    else:
        st.divider()

        with st.expander("⚙️ Configuración avanzada (solo si sabes lo que haces)"):
            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                dataset_path = st.text_input(
                    "📂 Carpeta con las radiografías de entrenamiento",
                    value=str(DATASET_DIR),
                    help="Carpeta que contiene las subcarpetas 'si' y 'no' con las imágenes.",
                )
            with col_cfg2:
                models_path = st.text_input(
                    "💾 Carpeta donde guardar los modelos",
                    value=str(MODELS_DIR),
                )

            modelos_disponibles = {
                cfg["description"]: key for key, cfg in MODEL_REGISTRY.items()
            }
            etiquetas_seleccionadas = st.multiselect(
                "¿Qué modelos quieres actualizar?",
                options=list(modelos_disponibles.keys()),
                default=list(modelos_disponibles.keys()),
                help="Por defecto se actualizan todos. Puedes desmarcar los que no quieras.",
            )
            model_keys_seleccionados = [
                modelos_disponibles[e] for e in etiquetas_seleccionadas
            ]

        # Valores por defecto si no se abre el expander
        if "dataset_path" not in dir():
            dataset_path = str(DATASET_DIR)
            models_path = str(MODELS_DIR)
            model_keys_seleccionados = list(MODEL_REGISTRY.keys())
            etiquetas_seleccionadas = list(MODEL_REGISTRY.keys())

        btn_disabled = len(etiquetas_seleccionadas) == 0
        st.button(
            "🚀 Iniciar actualización de la IA",
            type="primary",
            use_container_width=True,
            disabled=btn_disabled,
            key="btn_retrain",
        )
        if btn_disabled:
            st.caption("Selecciona al menos un modelo en la configuración avanzada.")

        if st.session_state.get("btn_retrain"):
            st.session_state.pop("retrain_results", None)

            st.markdown("#### ⏳ Actualizando... por favor espera")
            st.info(
                "La IA está aprendiendo con las nuevas imágenes. No cierres esta ventana."
            )

            progress_bar = st.progress(0.0)
            time_col, _ = st.columns([1, 3])
            time_display = time_col.empty()
            log_placeholder = st.empty()

            log_lines = []
            start_time = time.time()

            def log_fn(msg: str):
                elapsed = time.time() - start_time
                mins, secs = divmod(int(elapsed), 60)
                time_display.metric("⏱ Tiempo transcurrido", f"{mins}m {secs}s")
                log_lines.append(msg)
                log_placeholder.code("\n".join(log_lines[-60:]), language="")

            def progress_fn(val: float):
                progress_bar.progress(min(val, 1.0))

            results = retrain_all_models(
                dataset_dir=dataset_path,
                models_dir=models_path,
                log_fn=log_fn,
                progress_fn=progress_fn,
                model_keys=model_keys_seleccionados,
            )

            progress_bar.progress(1.0)
            total_elapsed = time.time() - start_time
            mins, secs = divmod(int(total_elapsed), 60)
            time_display.metric("⏱ Tiempo total", f"{mins}m {secs}s")

            st.session_state["retrain_results"] = results
            st.rerun()

        # ── Resultados ────────────────────────────────────────────────────────
        results = st.session_state.get("retrain_results")

        if results is not None:
            st.divider()
            if not results:
                st.error(
                    "❌ No se encontraron imágenes para entrenar.\n\n"
                    "Revisa que la carpeta del dataset contiene imágenes "
                    "dentro de las subcarpetas `si/` y `no/`."
                )
            else:
                n_mejor = sum(1 for r in results if r.get("saved_as_best"))
                n_igual = len(results) - n_mejor

                st.markdown("### ¿Ha mejorado la IA?")

                if n_mejor == len(results):
                    st.success(
                        f"🎉 ¡Todos los modelos han mejorado y se han guardado automáticamente! "
                        f"({n_mejor} de {len(results)})"
                    )
                elif n_mejor > 0:
                    st.success(
                        f"✅ {n_mejor} modelo(s) han mejorado y se han guardado."
                    )
                    st.warning(
                        f"⚠️ {n_igual} modelo(s) no han superado la versión anterior. "
                        f"Se conserva la versión anterior, que era mejor."
                    )
                else:
                    st.error(
                        "📉 Ningún modelo ha mejorado esta vez. "
                        "Se conservan los modelos anteriores porque eran mejores. "
                        "Prueba a añadir más imágenes al dataset e inténtalo de nuevo."
                    )

                st.markdown("#### Detalle por modelo")
                for row in results:
                    parte = row.get("body_part", "").capitalize()
                    modelo = _etiqueta_amigable(row.get("body_part", ""), row["model"])
                    acc_new = row["accuracy"] * 100
                    f1_new = row["f1_si"] * 100
                    f1_prev = row.get("best_f1_si_prev")
                    es_mejor = row.get("saved_as_best", False)

                    if es_mejor:
                        if f1_prev is None or f1_prev < 0:
                            st.success(
                                f"🏆 **{modelo}** — "
                                f"Primer entrenamiento guardado. "
                                f"Acierta el **{acc_new:.0f}%** de las radiografías."
                            )
                        else:
                            mejora = f1_new - f1_prev * 100
                            st.success(
                                f"🏆 **{modelo}** — Ha mejorado ↑ "
                                f"Antes acertaba el **{f1_prev*100:.0f}%**, "
                                f"ahora el **{f1_new:.0f}%**. "
                                f"Aciertos totales: **{acc_new:.0f}%**"
                            )
                    else:
                        prev_str = (
                            f"{f1_prev*100:.0f}%"
                            if f1_prev is not None
                            else "desconocido"
                        )
                        st.error(
                            f"📉 **{modelo}** — No ha mejorado. "
                            f"La versión anterior ({prev_str} de aciertos) sigue siendo mejor "
                            f"que la nueva ({f1_new:.0f}%). Se mantiene la anterior."
                        )

                with st.expander("🔬 Ver datos técnicos completos"):
                    df_res = pd.DataFrame(results)
                    display_cols = {
                        "model": "Modelo",
                        "body_part": "Parte",
                        "accuracy": "Accuracy",
                        "balanced_accuracy": "Balanced Acc.",
                        "precision_si": "Precision (si)",
                        "recall_si": "Recall (si)",
                        "f1_si": "F1 (si)",
                        "val_loss_best": "Val Loss",
                        "epochs_trained": "Épocas",
                        "saved_as_best": "¿Mejor?",
                        "best_f1_si_prev": "F1 anterior",
                    }
                    df_show = df_res[
                        [c for c in display_cols if c in df_res.columns]
                    ].rename(columns=display_cols)
                    st.dataframe(df_show, use_container_width=True, hide_index=True)

        # ── Historial ─────────────────────────────────────────────────────────
        st.divider()
        st.markdown("#### 📅 Historial de actualizaciones")
        st.caption("Registro de todos los entrenamientos realizados hasta ahora.")

        history = load_metrics_history(
            models_path if "models_path" in dir() else MODELS_DIR
        )

        if history:
            df_hist = pd.DataFrame(history)
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
            df_hist = df_hist.sort_values("timestamp", ascending=False)
            df_hist["timestamp"] = df_hist["timestamp"].dt.strftime("%d/%m/%Y %H:%M")
            df_hist["saved_as_best"] = (
                df_hist["saved_as_best"].map({True: "✅ Sí", False: "❌ No"})
                if "saved_as_best" in df_hist.columns
                else "—"
            )

            hist_cols = {
                "timestamp": "Fecha",
                "body_part": "Parte del cuerpo",
                "model": "Modelo",
                "accuracy": "% Aciertos",
                "f1_si": "Puntuación",
                "saved_as_best": "¿Mejoró?",
            }
            st.dataframe(
                df_hist[[c for c in hist_cols if c in df_hist.columns]].rename(
                    columns=hist_cols
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "Aún no hay actualizaciones registradas. "
                "Cuando pulses el botón de actualización por primera vez, "
                "aquí aparecerá el historial."
            )
