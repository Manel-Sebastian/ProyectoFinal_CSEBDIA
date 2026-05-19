# Día 2 — Construcción del Agente Multi-Paso (5h)

**Grupo 5 — Agente multi-paso (MedSAM + CNN + Gemini)**  
**Proyecto**: Clasificación de posicionamiento de dedo en escáner biométrico  
**Fecha**: Mayo 2026

---

## Objetivo del día

Implementar en código el agente diseñado en el Día 1 y verificar que el pipeline completo funciona de extremo a extremo sobre imágenes reales del dataset. No se toma ninguna decisión de arquitectura nueva — todo viene del diseño del Día 1.

El flujo implementado:

```
Imagen PNG
    │
    ▼  PASO 1 — MedSAM
    │  segmenta el dedo → máscara + score
    │  si score < 0.60 → ERROR (imagen no válida)
    │
    ▼  PASO 2 — ResidualCNN
    │  clasifica SI/NO sobre imagen enmascarada → clase + p_cnn
    │  si p_cnn >= 0.70 → VEREDICTO DIRECTO (CNN segura)
    │
    ▼  PASO 3 — Gemini (solo si p_cnn < 0.70)
       árbitro multimodal → clase_final + razón en texto
```

---

## Configuración global

Antes de cargar ningún modelo se resuelve la raíz del proyecto de forma portátil. Jupyter puede lanzarse desde cualquier directorio, así que no se puede usar `Path(".")`:

```python
def _find_project_root():
    current = Path(os.getcwd()).resolve()
    for p in [current] + list(current.parents):
        if (p / ".git").exists():
            return p
    return current

PROJECT_ROOT = _find_project_root()
```

Desde ahí se construyen todas las rutas:

```python
MEDSAM_CKPT = PROJECT_ROOT / "MedSAM_GeminiAPI/Dia1/medsam_vit_b.pth"
CNN_CKPT    = PROJECT_ROOT / "CNN_desde_0/modelosEntrenados/best_ResidualCNN.pth"
DATASET_DIR = PROJECT_ROOT / "CNN_desde_0/imagenes_procesadas"
```

Y los umbrales del Día 1:

```python
UMBRAL_CNN    = 0.70   # por debajo → Gemini arbitra
UMBRAL_MEDSAM = 0.60   # por debajo → imagen inválida
IDX_TO_LABEL  = {0: "no", 1: "si"}
GEMINI_MODEL  = "google/gemini-2.0-flash-lite-001"
```

---

## Tarea 1 + 2 — Implementación e integración *(2.5h)*

### Paso 1 — Carga de MedSAM (ViT-B)

MedSAM es el modelo SAM de Meta adaptado para imágenes médicas. El checkpoint puede estar guardado de dos formas distintas (con `model` como clave o directamente como `state_dict`), así que la carga lo detecta automáticamente:

```python
raw   = torch.load(MEDSAM_CKPT, map_location=DEVICE)
sam   = sam_model_registry["vit_b"](checkpoint=None)
state = raw.get("model", raw) if isinstance(raw, dict) else raw
sam.load_state_dict(state, strict=False)
sam.to(DEVICE).eval()
predictor = SamPredictor(sam)
```

`SamPredictor` es la interfaz de alto nivel de SAM: recibe la imagen una sola vez con `set_image()` y luego acepta múltiples prompts (bounding boxes, puntos) sin volver a procesar el backbone.

- **Parámetros**: ~90M (backbone ViT-B)
- **Entrada**: imagen RGB cualquier tamaño
- **Salida**: máscara binaria booleana + score de confianza (0–1)

---

### Paso 2 — Arquitectura ResidualCNN

Réplica exacta del modelo entrenado en `CNN_Main_Final.ipynb` del Grupo 2. Es imprescindible que sea idéntica para que `load_state_dict` funcione sin errores.

#### ResidualBlock

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        self.conv1      = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1        = nn.BatchNorm2d(out_channels)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2        = nn.BatchNorm2d(out_channels)
        self.downsample = downsample   # None si no cambia dimensión

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)   # ajusta canales/resolución para el skip
        return self.relu(out + identity)    # suma skip-connection + activación final
```

El `downsample` es la parte crítica: cuando `stride > 1` o los canales cambian, el tensor `identity` no tiene el mismo shape que `out`, y la suma fallaría. El `downsample` lo corrige con una convolución 1×1.

#### ResidualCNN completa

```
Entrada: 3 × 224 × 224

Conv2d(3→32, 7×7, stride=2) + BN + ReLU + MaxPool(3×3, stride=2)
    → 32 × 56 × 56

layer1: ResidualBlock(32→64) × 2         → 64 × 56 × 56
layer2: ResidualBlock(64→128, stride=2) × 2  → 128 × 28 × 28
layer3: ResidualBlock(128→256, stride=2) × 2 → 256 × 14 × 14

AdaptiveAvgPool2d(1×1)   → 256 × 1 × 1
Flatten                  → 256
Linear(256→128) + BN + ReLU + Dropout(0.4)
Linear(128→2)            → logits [NO, SI]
```

- **Parámetros**: 2,793,474 (~2.8M)
- **Salida**: logits → softmax → probabilidades [p_NO, p_SI]

La transformación de entrada debe coincidir exactamente con la del entrenamiento:

```python
CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
])
```

Si se usa una normalización diferente, las probabilidades de salida no tienen ningún sentido aunque el modelo cargue sin errores.

---

### Paso 3 — Cliente Gemini via OpenRouter

La API key se carga del fichero `.env` buscándolo hacia arriba en el árbol de directorios (igual que la raíz del proyecto):

```python
load_dotenv(find_dotenv())
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
```

Se usa el cliente `openai` porque OpenRouter implementa la misma API REST — no hace falta ningún SDK adicional de Google. Se verifica la conexión con un ping antes de continuar.

---

### Clase `AgenteMedico`

Cinco métodos, uno por responsabilidad:

#### `segmentar(img_array)` — Paso 1

```python
def segmentar(self, img_array):
    h, w = img_array.shape[:2]
    m    = 0.10
    bbox = np.array([[int(w*m), int(h*m), int(w*(1-m)), int(h*(1-m))]])
    self.predictor.set_image(img_array)
    masks, scores, _ = self.predictor.predict(box=bbox, multimask_output=True)
    idx   = int(np.argmax(scores))
    score = float(scores[idx])
    if score < self.UMBRAL_MEDSAM:
        raise ValueError(f"Segmentación insuficiente: score={score:.3f} < {self.UMBRAL_MEDSAM}")
    return masks[idx], score
```

El bounding box es el 80% central de la imagen (margen 10% por cada lado). `multimask_output=True` hace que SAM proponga 3 máscaras candidatas — se elige la de mayor score. Si ninguna supera 0.60, se lanza `ValueError` y el agente se detiene.

#### `clasificar(img_array, mask)` — Paso 2

```python
def clasificar(self, img_array, mask):
    img_masked = img_array.copy()
    img_masked[~mask] = 0                  # píxeles fuera de máscara → negro

    tensor = CNN_TRANSFORM(
        Image.fromarray(img_masked.astype(np.uint8))
    ).unsqueeze(0).to(self.device)

    with torch.no_grad():
        probs = torch.softmax(self.cnn(tensor), dim=1).squeeze().cpu().numpy()

    idx_pred = int(np.argmax(probs))
    clase    = IDX_TO_LABEL[idx_pred].upper()
    return clase, float(probs[idx_pred])
```

Poner a negro los píxeles fuera de la máscara (en lugar de recortar al bbox) mantiene la resolución 224×224 completa y preserva el contexto espacial — la CNN ve dónde está el dedo dentro del escáner.

#### `necesita_arbitraje(p_cnn)`

```python
def necesita_arbitraje(self, p_cnn):
    return p_cnn < self.UMBRAL_CNN   # True → activar Gemini
```

Lógica del Día 1: Gemini (70% accuracy) es mejor árbitro que la CNN (60%) cuando `p_cnn < 0.70`.

#### `consultar_gemini(img_array, mask, pred_cnn, p_cnn)` — Paso 3

Gemini recibe dos imágenes en base64 + texto:

```python
overlay       = img_array.copy()
overlay[mask] = [50, 220, 80]    # máscara MedSAM coloreada en verde
```

Se envía la imagen original (Gemini ve el escáner completo) y el overlay verde (Gemini sabe exactamente qué zona detectó MedSAM como dedo). Esto es más informativo que una máscara binaria blanco/negro.

**Prompt de Gemini** (versión final tras iteraciones):

```
Eres un sistema experto en análisis de posicionamiento biométrico.
Se te presentan dos imágenes de una radiografía térmica de un dedo sobre un escáner biométrico.

IMAGEN 1: Fotografía original del dedo sobre el escáner.
IMAGEN 2: Máscara de segmentación de MedSAM (región del dedo resaltada en verde).

El clasificador CNN predijo: [SI/NO] con confianza [XX%].
Esta confianza es BAJA (< 70%), necesitamos tu criterio.

El escáner tiene una franja luminosa (zona brillante). El dedo entero debe quedar centrado
encima de esa franja, sin desplazarse ni salirse. Ese es el único criterio.

CORRECTO (SI):
- El dedo está colocado encima de la franja luminosa y centrado en ella.
- El dedo no se sale de la franja ni por los lados ni por arriba/abajo.
- El dedo cubre la franja de forma plana, sin estar girado ni inclinado.

INCORRECTO (NO):
- El dedo está desplazado: parte del dedo queda fuera de la franja luminosa.
- El dedo solo toca un extremo o un lado de la franja, no el centro.
- El dedo está girado o inclinado de modo que no cubre bien la franja.

Responde ÚNICAMENTE con este formato exacto:
VEREDICTO: [SI / NO]
RAZÓN: [Una sola frase con el criterio visual determinante]
```

> **Por qué este prompt y no otro**: versiones anteriores mencionaban "yema del dedo", lo que hacía que Gemini se fijara solo en la punta en lugar del dedo completo, empeorando el accuracy. La versión final habla siempre de "el dedo" como unidad y de "la franja luminosa" como referencia, que es exactamente lo que el pipeline de Oscar detecta con `detect_light_window`.

El parsing extrae `VEREDICTO:` y `RAZÓN:` buscando por líneas, sin depender de posición exacta:

```python
for line in raw.splitlines():
    upper = line.upper()
    if upper.startswith("VEREDICTO:"):
        clase_final = "SI" if "SI" in upper else "NO"
    elif upper.startswith("RAZ"):
        razon = line.split(":", 1)[-1].strip()
```

Se llama con `temperature=0.1` para minimizar variabilidad — Gemini como árbitro debe ser determinista.

#### `veredicto_final(ruta_imagen)` — Orquestador

Ejecuta los tres pasos y devuelve un diccionario con trazabilidad completa:

```python
{
    "imagen"          : "nombre.png",
    "error"           : None,        # o mensaje si score_medsam < 0.60
    "score_medsam"    : 0.6420,
    "clase_cnn"       : "SI",
    "p_cnn"           : 0.8134,
    "arbitraje_gemini": False,       # True si p_cnn < 0.70
    "clase_final"     : "SI",
    "razon_gemini"    : None,        # o texto si Gemini intervino
}
```

El campo `error` permite saber si la imagen fue rechazada en el Paso 1 sin que el código pete. El campo `arbitraje_gemini` permite contar cuántas imágenes pasaron por Gemini en el análisis del Día 3.

---

## Tarea 3 — Prueba end-to-end *(1h)*

### Ranking CNN sobre todo el dataset

Antes de los tests, se pasa la CNN (sin MedSAM) por las 70 imágenes para ordenarlas por `p_cnn`. Esto permite seleccionar los casos extremos de forma objetiva:

```python
all_imgs = sorted(train/si + train/no + test/si + test/no)   # 70 imágenes

ranking = []
for img_path in all_imgs:
    tensor = CNN_TRANSFORM(Image.open(img_path)).unsqueeze(0)
    probs  = softmax(model_cnn(tensor))
    ranking.append({"path": img_path, "p_cnn": probs.max(), ...})

ranking.sort(key=lambda r: r["p_cnn"], reverse=True)
```

### Distribución de confianza CNN

Se visualizó la distribución de `p_cnn` sobre las 70 imágenes:

- **Verde** (`p_cnn ≥ 0.70`): CNN decide directamente sin Gemini
- **Naranja** (`p_cnn < 0.70`): Gemini tiene que arbitrar
- Línea roja discontinua en el umbral 0.70

![Distribución confianza CNN](distribucion_confianza_cnn.png)

La distribución confirma que hay una porción significativa del dataset en zona de incertidumbre, validando la necesidad del árbitro.

### Test A — 5 imágenes de alta confianza

Las 5 imágenes con mayor `p_cnn` del ranking:

- **Resultado esperado**: Gemini no se activa en ningún caso
- **Resultado obtenido**: el agente resuelve los 5 directamente con la CNN, sin ninguna llamada a la API

Esto es importante: en casos donde la CNN es segura, el agente es rápido y barato (sin latencia de red ni coste de API).

### Test B — 5 imágenes de baja confianza

Las 5 imágenes con menor `p_cnn` del ranking:

- **Resultado esperado**: Gemini interviene en los 5 casos
- **Resultado obtenido**: Gemini arbitra los 5, devolviendo siempre `VEREDICTO:` y `RAZÓN:` correctamente formateados

Valida que el prompt produce salida parseable de forma consistente.

---

## Tarea 4 — Visualización del pipeline Oscar + MedSAM *(1h)*

### Motivación

Para hacer el bounding box de MedSAM más preciso, se integró el pipeline de preprocesado OpenCV de Oscar (`CNN_desde_0/CNNs_Oscar/dia4/day1_opencv_helpers.py`). En el agente principal (`segmentar()`) se usa un bbox conservador del 80% central — funciona siempre pero no es óptimo. La visualización muestra el pipeline completo con el bbox preciso.

### Flujo completo de detección del dedo

```
Imagen original (RGB)
    │
    ▼  normalize_with_white_background()
    │  Remapeo lineal de brillo: fondo → gris 200, panel → gris 25
    │  Hace el panel negro más consistente para detectarlo
    │
    ▼  detect_black_panel()  →  quad (4 esquinas del panel, coords imagen)
    │  Detección del rectángulo negro del escáner por umbral de oscuridad
    │  Devuelve el polígono en coordenadas de la imagen original
    │
    ▼  warp_panel(quad)  →  panel (640×480) + warp_matrix
    │  Perspectiva corregida: el panel queda frontal y rectangular
    │  warp_matrix permite proyectar coords de panel ↔ imagen original
    │
    ├─▶  segment_hand()  →  hand_mask (640×480, bool)
    │    Segmenta la mano por color (YCrCb skin detection) + brillo
    │    Excluye zonas muy brillantes (la luz del escáner)
    │
    ├─▶  detect_light_window()  →  light_mask + light_box
    │    Detecta la franja luminosa del escáner
    │    Criterio: baja saturación + alto brillo (luz blanca del sensor)
    │
    ▼  Intersección: binary_dilation(hand_mask, 40px) ∩ light_mask
    │  = zona exacta donde el dedo toca la luz del escáner
    │  La dilatación es necesaria: la mano no llega literalmente a los píxeles de luz
    │
    ▼  cv2.perspectiveTransform(contact_box, inv(warp_matrix))
    │  Proyección inversa: coords del panel → coords de la imagen original
    │
    ▼  MedSAM.predict(box=bbox_finger, point_coords=[center], point_labels=[1])
       bbox preciso del dedo + punto central como foreground hint
       → máscara final del dedo mucho más ajustada
```

#### Por qué la intersección mano∩luz

El problema con usar solo la `light_box` era que a veces la zona luminosa es grande (toda la franja del escáner) y el bbox proyectado cogía más de lo que era el dedo. Al intersectar con la `hand_mask` dilatada, se recorta la zona de luz a la parte que efectivamente tiene mano encima — que es exactamente el dedo sobre el escáner.

```python
hand_dilated = binary_dilation(hand_mask_p, structure=np.ones((40, 40), bool))
contact_mask = light_mask_panel & hand_dilated
```

La dilatación de 40px es necesaria porque `segment_hand` excluye los píxeles más brillantes (la propia luz), así que la mano no llega hasta el borde de la zona luminosa. La dilatación cierra ese hueco.

#### Por qué el punto foreground en MedSAM

SAM con solo un bounding box a veces segmenta el objeto más grande dentro del bbox, que puede no ser el dedo. Al añadir el centro del bbox como punto de foreground (`point_labels=[1]`), se le dice explícitamente a SAM "el objeto que me interesa está aquí", reduciendo falsos positivos:

```python
masks, scores, _ = agente.predictor.predict(
    point_coords    = np.array([[cx, cy]]),
    point_labels    = np.array([1]),      # 1 = foreground
    box             = bbox_finger,
    multimask_output= True,
)
```

### Las 6 columnas de la visualización

| Columna | Qué muestra |
|---------|-------------|
| ① Original + bbox (naranja) | El prompt exacto que recibe MedSAM. Punto rojo = foreground hint |
| ② Normalización + panel | Brillo homogéneo. Polígono rojo = esquinas del panel detectadas por Oscar |
| ③ Panel rectificado + mano | Panel en perspectiva frontal. Verde = mano detectada por YCrCb |
| ④ Zona de contacto (mano∩luz) | Naranja = donde el dedo toca la franja luminosa. Esta es la región que se proyecta como bbox |
| ⑤ MedSAM — Máscara final | Verde = segmentación del dedo. Score MedSAM en el eje x |
| ⑥ Input CNN | Lo que realmente ve el clasificador: imagen enmascarada (fondo negro) |

![Pipeline Oscar + MedSAM](oscar_medsam_pipeline_dia2.png)

### Pipeline end-to-end visualizado

Para las 10 imágenes del Test A + B se muestra el veredicto completo del agente en 4 columnas:

| Columna | Contenido |
|---------|-----------|
| Original + bbox | Imagen con el prompt de MedSAM |
| MedSAM overlay | Máscara verde del dedo |
| CNN input | Imagen enmascarada que entra al clasificador |
| Veredicto | Clase final, p_cnn, si Gemini intervino y su razón |

- **Borde azul**: la CNN tomó la decisión directamente (`p_cnn ≥ 0.70`)
- **Borde naranja**: Gemini intervino como árbitro (`p_cnn < 0.70`)

![Pipeline completo](pipeline_completo_dia2.png)

---

## Decisiones de implementación y por qué

| Decisión | Alternativa descartada | Razón |
|----------|----------------------|-------|
| Máscara aplicada poniendo fondo a negro | Recortar imagen al bbox | Mantiene resolución 224×224 y contexto espacial del dedo dentro del escáner |
| Overlay verde para Gemini (no máscara binaria) | Imagen B/W de la máscara | El color aporta más información al modelo multimodal que un mapa binario |
| Parsing por `startswith("VEREDICTO:")` | JSON estructurado | Gemini a veces añade texto antes/después del formato pedido; el parseo por líneas es más robusto |
| `temperature=0.1` en Gemini | Temperatura por defecto | El árbitro debe ser lo más determinista posible, no creativo |
| Prompt sin "yema" — habla de "el dedo" | Prompt con "yema del dedo" | Al mencionar "yema", Gemini se fijaba solo en la punta y empeoraba el accuracy |
| Punto foreground + bbox en MedSAM | Solo bbox | SAM con punto explícito es más preciso cuando el bbox todavía tiene algo de margen |
| Dict de trazabilidad completo | Solo devolver la clase | Permite auditar cada paso en el Día 3 sin reejecutar el agente |

---

## Resumen del Día 2

| Tarea | Tiempo | Resultado |
|-------|--------|-----------|
| Configuración + carga de modelos | 0.5h | ✅ MedSAM ViT-B, ResidualCNN, Gemini vía OpenRouter |
| Implementación `AgenteMedico` (5 métodos) | 2h | ✅ Pipeline MedSAM → CNN → Gemini funcional |
| Prueba end-to-end Test A + B | 1h | ✅ Alta confianza: CNN directa. Baja confianza: Gemini arbitra |
| Visualización Oscar + MedSAM (6 pasos) | 1h | ✅ Bbox preciso por intersección mano∩luz + foreground hint |
| Iteraciones del prompt de Gemini | 0.5h | ✅ Eliminado "yema", criterio centrado en la franja luminosa |

**Total**: 5h

### Ficheros generados

| Fichero | Contenido |
|---------|-----------|
| `agente_medico_dia2.ipynb` | Notebook completo con toda la implementación |
| `distribucion_confianza_cnn.png` | Distribución de p_cnn sobre 70 imágenes |
| `oscar_medsam_pipeline_dia2.png` | Pipeline Oscar + MedSAM (6 columnas × 6 imágenes) |
| `pipeline_completo_dia2.png` | Pipeline end-to-end para Test A + B |

---

*Documentación del Día 2 — Grupo 5 — Proyecto IA RX — Mayo 2026*
