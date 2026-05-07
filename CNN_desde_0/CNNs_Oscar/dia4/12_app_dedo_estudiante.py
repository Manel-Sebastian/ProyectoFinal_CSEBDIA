# Escribe aquí tu app de Streamlit.
# App Streamlit: Detector de postura del dedo
# Heurística OpenCV (detección de piel + análisis de contornos)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Librerías ───────────────────────────────────────────────────────────────
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ── 2. Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Detector de postura del dedo",
    page_icon="🫵",
    layout="centered",
)

# ── 3. Funciones auxiliares ────────────────────────────────────────────────────

def preprocess_image(img_array: np.ndarray) -> np.ndarray:
    """Redimensiona y convierte a BGR."""
    img = cv2.resize(img_array, (224, 224))
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def get_skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Máscara binaria de piel (HSV ∩ YCrCb)."""
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    mask_hsv   = cv2.inRange(hsv,
                             np.array([0, 20, 70],    dtype=np.uint8),
                             np.array([25, 255, 255], dtype=np.uint8))
    mask_ycrcb = cv2.inRange(ycrcb,
                              np.array([0, 133, 77],   dtype=np.uint8),
                              np.array([255, 173, 127], dtype=np.uint8))

    mask   = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


@st.cache_data(show_spinner=False)
def predict_dedo(img_bytes: bytes) -> dict:
    """
    Pipeline de inferencia completo.
    Recibe los bytes de la imagen para que st.cache_data pueda hashearlos.
    """
    pil_image = Image.open(img_bytes).convert("RGB")
    img_array = np.array(pil_image)
    bgr  = preprocess_image(img_array)
    mask = get_skin_mask(bgr)

    total_px   = mask.size
    skin_ratio = np.sum(mask > 0) / total_px

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aspect_ratio = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / (h + 1e-6)

    skin_ok  = 0.05 < skin_ratio < 0.45
    shape_ok = aspect_ratio < 0.7

    skin_score  = float(np.clip(1.0 - abs(skin_ratio - 0.15) / 0.30, 0, 1))
    shape_score = float(np.clip(1.0 - aspect_ratio, 0, 1))
    confidence  = round(0.5 * skin_score + 0.5 * shape_score, 2)

    label = 'sí' if (skin_ok and shape_ok) else 'no'

    if label == 'no':
        if skin_ratio < 0.05:
            message = '⚠️ No se detecta suficiente piel. Acerca el dedo a la cámara.'
        elif skin_ratio >= 0.45:
            message = '⚠️ Demasiado fondo de piel. Aleja la mano o mejora el fondo.'
        else:
            message = '⚠️ El dedo parece horizontal. Colócalo verticalmente.'
    else:
        message = '✅ ¡Postura correcta!'

    return {
        'label':       label,
        'confidence':  confidence,
        'message':     message,
        'mask_img':    mask,
        'pil_image':   pil_image,
    }


# ── 4. Interfaz ────────────────────────────────────────────────────────────────

st.title("🫵 Detector de postura del dedo")
st.markdown(
    "Sube una imagen de tu mano y la app te dirá si el dedo está correctamente "
    "posicionado. Usa la heurística de detección de piel del **Día 1**."
)
st.divider()

uploaded_file = st.file_uploader(
    "Sube una imagen (JPG, PNG…)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:

    # ── Inferencia ────────────────────────────────────────────────────────────
    with st.spinner("Analizando imagen…"):
        result = predict_dedo(uploaded_file)

    # ── Visualización: dos columnas ───────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen original")
        st.image(result['pil_image'], use_column_width=True)
    with col2:
        st.subheader("Máscara de piel")
        st.image(result['mask_img'], use_column_width=True, clamp=True)

    st.divider()

    # ── Resultado ─────────────────────────────────────────────────────────────
    label = result['label']
    color = "green" if label == 'sí' else "red"
    emoji = "✅" if label == 'sí' else "❌"

    st.markdown(
        f"### Predicción: :{color}[{emoji} **{label.upper()}**]"
    )

    st.metric(
        label="Confianza",
        value=f"{result['confidence']:.0%}",
    )
    st.progress(result['confidence'])

    st.info(result['message'])

else:
    st.markdown(
        "👆 **Sube una imagen** para comenzar el análisis."
    )
