"""Genera agente_medico_dia3.ipynb leyendo las celdas base del Dia2."""
import json, sys
from pathlib import Path

DIA2 = Path(__file__).parent.parent / "Dia2" / "agente_medico_dia2.ipynb"
OUT  = Path(__file__).parent / "agente_medico_dia3.ipynb"

with open(DIA2) as f:
    nb2 = json.load(f)

def code(src):   return {"cell_type": "code",     "metadata": {}, "source": [src], "outputs": [], "execution_count": None}
def md(src):     return {"cell_type": "markdown",  "metadata": {}, "source": [src]}

# ── células base copiadas de Dia2 (índices exactos) ──────────────────────────
src_imports   = "".join(nb2["cells"][2]["source"])
src_config    = "".join(nb2["cells"][4]["source"])
src_medsam    = "".join(nb2["cells"][6]["source"])
src_cnn_arch  = "".join(nb2["cells"][8]["source"])
src_gemini    = "".join(nb2["cells"][10]["source"])
src_agente    = "".join(nb2["cells"][12]["source"])
src_instancia = "".join(nb2["cells"][14]["source"])

# Añadir sklearn a imports
src_imports = src_imports.replace(
    "import matplotlib.patches as mpatches",
    "import matplotlib.patches as mpatches\nfrom sklearn.metrics import (accuracy_score, precision_score,\n"
    "    recall_score, f1_score, confusion_matrix, classification_report)\nimport warnings\nwarnings.filterwarnings('ignore')"
)

# ── código Día 3 ──────────────────────────────────────────────────────────────

TAREA1_RUN = '''\
# Set de test completo
test_si  = sorted((DATASET_DIR / "test" / "si").glob("*.png"))
test_no  = sorted((DATASET_DIR / "test" / "no").glob("*.png"))
test_imgs = [(p, "SI") for p in test_si] + [(p, "NO") for p in test_no]

print(f"Set de test: {len(test_si)} SI + {len(test_no)} NO = {len(test_imgs)} imágenes")
print("Ejecutando el agente completo (MedSAM → CNN → Gemini)...\\n")

resultados = []
for i, (img_path, clase_real) in enumerate(test_imgs):
    res = agente.veredicto_final(img_path)
    res["clase_real"] = clase_real

    if res["error"]:
        estado = f"✗ ERROR: {res['error']}"
    else:
        decide  = "[Gemini]" if res["arbitraje_gemini"] else "[CNN]   "
        acierto = "✓" if res["clase_final"] == clase_real else "✗"
        estado  = f"{decide} → {res['clase_final']} {acierto}"

    print(f"  [{i+1:2d}/{len(test_imgs)}] {img_path.name:28s} | real: {clase_real} | {estado}")
    resultados.append(res)

print(f"\\n✓ Evaluación completada: {len(resultados)} imágenes procesadas")
'''

TAREA1_STATS = '''\
validos = [r for r in resultados if not r["error"]]
errores = [r for r in resultados if r["error"]]

n_total  = len(validos)
n_gemini = sum(1 for r in validos if r["arbitraje_gemini"])
n_cnn    = n_total - n_gemini
pct_g    = 100 * n_gemini / n_total if n_total else 0

print(f"Imágenes válidas        : {n_total}")
print(f"Rechazadas (MedSAM<0.60): {len(errores)}")
print(f"CNN decide directamente : {n_cnn}  ({100 - pct_g:.1f}%)")
print(f"Gemini interviene       : {n_gemini}  ({pct_g:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart — quién decide
axes[0].pie(
    [n_cnn, n_gemini],
    labels=[f"CNN directa\\n({n_cnn} imgs)", f"Gemini árbitro\\n({n_gemini} imgs)"],
    colors=["#27ae60", "#f39c12"],
    autopct="%1.1f%%", startangle=90,
    textprops={"fontsize": 12},
)
axes[0].set_title("¿Quién decide en el set de test?", fontsize=12, fontweight="bold")

# Distribución de score MedSAM
scores = [r["score_medsam"] for r in validos]
axes[1].hist(scores, bins=12, color="#8e44ad", alpha=0.8, edgecolor="white")
axes[1].axvline(UMBRAL_MEDSAM, color="red", linestyle="--", linewidth=1.5, label=f"Umbral MedSAM ({UMBRAL_MEDSAM})")
axes[1].set_xlabel("Score MedSAM", fontsize=11)
axes[1].set_ylabel("Nº imágenes", fontsize=11)
axes[1].set_title("Distribución de confianza MedSAM", fontsize=12, fontweight="bold")
axes[1].legend()

plt.suptitle("Tarea 1 — Ejecución sobre el set de test completo", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("dia3_intervencion_gemini.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ dia3_intervencion_gemini.png guardada")
'''

TAREA2_CNN = '''\
# CNN sola: sin máscara, sin Gemini — imagen original directa
print("Evaluando CNN sola sobre el set de test (sin máscara, sin Gemini)...")

cnn_sola = []
model_cnn.eval()
for img_path, clase_real in test_imgs:
    tensor = CNN_TRANSFORM(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model_cnn(tensor), dim=1).squeeze().cpu().numpy()
    pred = IDX_TO_LABEL[int(np.argmax(probs))].upper()
    cnn_sola.append({
        "imagen": img_path.name, "clase_real": clase_real,
        "pred": pred, "p_cnn": float(probs.max()),
    })
    acierto = "✓" if pred == clase_real else "✗"
    print(f"  {img_path.name:28s} | real: {clase_real} | CNN: {pred} ({probs.max():.2f}) {acierto}")

acc_cnn = accuracy_score([r["clase_real"] for r in cnn_sola], [r["pred"] for r in cnn_sola])
print(f"\\nCNN sola — Accuracy: {acc_cnn:.1%}")
'''

TAREA2_COMPARE = '''\
# Alinear imágenes válidas (excluir las que falló MedSAM)
nombres_validos = {r["imagen"] for r in validos}
cnn_f = [r for r in cnn_sola if r["imagen"] in nombres_validos]

y_true_cnn = [r["clase_real"] for r in cnn_f]
y_pred_cnn = [r["pred"] for r in cnn_f]

y_true_ag  = [r["clase_real"]  for r in validos]
y_pred_ag  = [r["clase_final"] for r in validos]

def metricas(yt, yp, nombre):
    acc   = accuracy_score(yt, yp)
    p_si  = precision_score(yt, yp, pos_label="SI", zero_division=0)
    r_si  = recall_score(yt, yp, pos_label="SI", zero_division=0)
    f1_si = f1_score(yt, yp, pos_label="SI", zero_division=0)
    f1_no = f1_score(yt, yp, pos_label="NO", zero_division=0)
    mf1   = f1_score(yt, yp, average="macro", zero_division=0)
    print(f"\\n{'='*10} {nombre} {'='*10}")
    print(f"  Accuracy       : {acc:.1%}")
    print(f"  Precision (SI) : {p_si:.1%}")
    print(f"  Recall    (SI) : {r_si:.1%}")
    print(f"  F1        (SI) : {f1_si:.1%}")
    print(f"  F1        (NO) : {f1_no:.1%}")
    print(f"  Macro F1       : {mf1:.1%}")
    return dict(acc=acc, p_si=p_si, r_si=r_si, f1_si=f1_si, f1_no=f1_no, mf1=mf1)

m_cnn = metricas(y_true_cnn, y_pred_cnn, "CNN sola")
m_ag  = metricas(y_true_ag,  y_pred_ag,  "Agente completo")

# ── Matrices de confusión lado a lado ────────────────────────────────────────
labels = ["NO", "SI"]
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (yt, yp, titulo, acc) in zip(axes, [
        (y_true_cnn, y_pred_cnn, "CNN sola",       m_cnn["acc"]),
        (y_true_ag,  y_pred_ag,  "Agente completo", m_ag["acc"]),
]):
    cm = confusion_matrix(yt, yp, labels=labels)
    ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Real",       fontsize=11)
    ax.set_title(f"{titulo}\\nAccuracy: {acc:.1%}", fontsize=12, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.suptitle("Comparativa: CNN sola vs Agente multi-paso", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("dia3_comparativa_confusion.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ dia3_comparativa_confusion.png guardada")
'''

TAREA2_BARS = '''\
etiquetas = ["Accuracy", "Precision\\n(SI)", "Recall\\n(SI)", "F1\\n(SI)", "F1\\n(NO)", "Macro F1"]
vals_cnn = [m_cnn["acc"], m_cnn["p_si"], m_cnn["r_si"], m_cnn["f1_si"], m_cnn["f1_no"], m_cnn["mf1"]]
vals_ag  = [m_ag["acc"],  m_ag["p_si"],  m_ag["r_si"],  m_ag["f1_si"],  m_ag["f1_no"],  m_ag["mf1"]]

x = np.arange(len(etiquetas))
w = 0.35
fig, ax = plt.subplots(figsize=(13, 5))
b1 = ax.bar(x - w/2, vals_cnn, w, label="CNN sola",        color="#2980b9", alpha=0.85)
b2 = ax.bar(x + w/2, vals_ag,  w, label="Agente completo", color="#27ae60", alpha=0.85)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.0%}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.axhline(0.70, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="70% referencia")
ax.set_xticks(x); ax.set_xticklabels(etiquetas, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Valor (0–1)", fontsize=11)
ax.set_title("CNN sola vs Agente multi-paso — Métricas comparativas", fontsize=13, fontweight="bold")
ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("dia3_comparativa_metricas.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ dia3_comparativa_metricas.png guardada")
'''

TAREA3_ANALYZE = '''\
# Filtrar solo intervenciones de Gemini
gemini_casos = [r for r in validos if r["arbitraje_gemini"]]

corrige    = []  # CNN mal, Gemini bien
empeora    = []  # CNN bien, Gemini mal
ambos_bien = []  # ambos aciertan
ambos_mal  = []  # ambos fallan

for r in gemini_casos:
    cnn_ok    = r["clase_cnn"]   == r["clase_real"]
    gemini_ok = r["clase_final"] == r["clase_real"]
    if   not cnn_ok and gemini_ok:  corrige.append(r)
    elif cnn_ok and not gemini_ok:  empeora.append(r)
    elif cnn_ok and gemini_ok:      ambos_bien.append(r)
    else:                           ambos_mal.append(r)

n_g = len(gemini_casos)
pct = lambda x: f"{100*len(x)/n_g:.0f}%" if n_g else "—"

print(f"Total intervenciones Gemini : {n_g}")
print(f"  ✅ Gemini CORRIGE a CNN   : {len(corrige):2d}  ({pct(corrige)})")
print(f"  ❌ Gemini EMPEORA a CNN   : {len(empeora):2d}  ({pct(empeora)})")
print(f"  🟢 Ambos CORRECTOS        : {len(ambos_bien):2d}  ({pct(ambos_bien)})")
print(f"  🔴 Ambos INCORRECTOS      : {len(ambos_mal):2d}  ({pct(ambos_mal)})")

# Gráfico de barras de categorías
cats   = ["Gemini corrige\\nCNN", "Gemini empeora\\nCNN", "Ambos\\ncorrectos", "Ambos\\nincorrectos"]
vals   = [len(corrige), len(empeora), len(ambos_bien), len(ambos_mal)]
colors = ["#27ae60", "#e74c3c", "#3498db", "#e67e22"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(cats, vals, color=colors, width=0.5, alpha=0.88)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            str(v), ha="center", va="bottom", fontsize=16, fontweight="bold")
ax.set_ylabel("Nº de imágenes", fontsize=11)
ax.set_title(f"Análisis de las {n_g} intervenciones de Gemini", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(vals) + 2 if vals else 3)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("dia3_analisis_gemini.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ dia3_analisis_gemini.png guardada")

# Imprimir razones de Gemini
for titulo, grupo in [
    ("GEMINI CORRIGE a CNN", corrige),
    ("GEMINI EMPEORA a CNN", empeora),
]:
    if grupo:
        print(f"\\n--- {titulo} ---")
        for r in grupo:
            print(f"  {r['imagen']:28s} | real: {r['clase_real']} | "
                  f"CNN: {r['clase_cnn']} ({r['p_cnn']:.2f}) → Gemini: {r['clase_final']}")
            print(f"    Razón: {r['razon_gemini']}")
'''

TAREA3_VIS = '''\
def _find_img(nombre):
    for split in ["train", "test"]:
        for clase in ["si", "no"]:
            p = DATASET_DIR / split / clase / nombre
            if p.exists():
                return p
    return None

def vis_casos(casos, titulo, color, max_show=5):
    if not casos:
        print(f"Sin casos: {titulo}")
        return
    n = min(len(casos), max_show)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, casos[:n]):
        img_path = _find_img(r["imagen"])
        if img_path:
            ax.imshow(np.array(Image.open(img_path).convert("RGB")))
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
        ax.set_title(
            f"Real: {r['clase_real']}\\n"
            f"CNN: {r['clase_cnn']} ({r['p_cnn']:.2f})\\n"
            f"Gemini: {r['clase_final']}",
            fontsize=8, fontweight="bold",
        )
        razon = (r["razon_gemini"] or "")[:55]
        ax.set_xlabel(razon, fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(titulo, fontsize=12, fontweight="bold", color=color)
    plt.tight_layout()
    fname = "dia3_" + titulo.lower().replace(" ", "_")[:35] + ".png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"✓ {fname} guardada")

vis_casos(corrige,    "Gemini CORRIGE a CNN",    "#27ae60")
vis_casos(empeora,    "Gemini EMPEORA a CNN",    "#e74c3c")
vis_casos(ambos_bien, "Ambos CORRECTOS",          "#3498db")
vis_casos(ambos_mal,  "Ambos INCORRECTOS",        "#e67e22")
'''

TAREA4_SIM = '''\
# Simulación de umbrales usando los datos ya recopilados.
# Para t <= 0.70: simulación exacta (tenemos respuesta Gemini para todos los p_cnn < 0.70).
# Para t > 0.70: los casos nuevos (0.70 <= p_cnn < t) no tienen respuesta Gemini → se usa CNN.

thresholds  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
sim_results = []

for t in thresholds:
    preds, trues, n_g_t = [], [], 0
    for r in validos:
        p = r["p_cnn"]
        if p >= t:
            pred = r["clase_cnn"]     # CNN decide
        elif r["arbitraje_gemini"]:   # p < t y Gemini fue llamado (p < 0.70 original)
            pred = r["clase_final"]
            n_g_t += 1
        else:                         # p < t pero p >= 0.70 → sin Gemini → usamos CNN
            pred = r["clase_cnn"]
        preds.append(pred)
        trues.append(r["clase_real"])

    acc  = accuracy_score(trues, preds)
    mf1  = f1_score(trues, preds, average="macro", zero_division=0)
    f1si = f1_score(trues, preds, pos_label="SI", zero_division=0)
    sim_results.append(dict(
        t=t, acc=acc, mf1=mf1, f1si=f1si,
        n_g=n_g_t, pct_g=100*n_g_t/len(validos),
        exacto=(t <= UMBRAL_CNN),
    ))

print(f"{'Umbral':>7} {'Accuracy':>10} {'Macro F1':>10} {'F1 SI':>8} {'%Gemini':>10}  Nota")
print("-" * 62)
for s in sim_results:
    nota = "simulación exacta" if s["exacto"] else "estimado (sin Gemini para casos nuevos)"
    print(f"  {s['t']:.2f}  {s['acc']:>10.1%} {s['mf1']:>10.1%} {s['f1si']:>8.1%} "
          f"{s['pct_g']:>9.1f}%  {nota}")
'''

TAREA4_PLOT = '''\
t_vals   = [s["t"]     for s in sim_results]
acc_vals = [s["acc"]   for s in sim_results]
f1_vals  = [s["mf1"]   for s in sim_results]
f1si_v   = [s["f1si"]  for s in sim_results]
pct_g    = [s["pct_g"] for s in sim_results]
exacto   = [s["exacto"]for s in sim_results]

# Separar exactos y estimados
t_ex   = [t for t, e in zip(t_vals, exacto) if e]
t_est  = [t for t, e in zip(t_vals, exacto) if not e]
acc_ex = [a for a, e in zip(acc_vals, exacto) if e]
acc_est= [a for a, e in zip(acc_vals, exacto) if not e]
f1_ex  = [f for f, e in zip(f1_vals, exacto) if e]
f1_est = [f for f, e in zip(f1_vals, exacto) if not e]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Plot 1: métricas vs umbral ────────────────────────────────────────────────
ax = axes[0]
ax.plot(t_ex,  acc_ex,  "o-", color="#2980b9", linewidth=2, label="Accuracy (exacto)")
ax.plot(t_ex,  f1_ex,   "s-", color="#27ae60", linewidth=2, label="Macro F1 (exacto)")
if t_est:
    ax.plot([t_ex[-1]] + t_est, [acc_ex[-1]] + acc_est, "o--", color="#2980b9", alpha=0.4, label="Accuracy (estimado)")
    ax.plot([t_ex[-1]] + t_est, [f1_ex[-1]]  + f1_est,  "s--", color="#27ae60", alpha=0.4, label="Macro F1 (estimado)")
ax.axvline(UMBRAL_CNN, color="red", linestyle="--", linewidth=1.5, label=f"Umbral actual ({UMBRAL_CNN})")
ax.set_xlabel("Umbral de confianza CNN", fontsize=11)
ax.set_ylabel("Métrica", fontsize=11)
ax.set_title("Accuracy y Macro F1 según umbral", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ── Plot 2: coste (% Gemini) vs umbral ───────────────────────────────────────
ax2 = axes[1]
ax2.plot(t_vals, pct_g, "D-", color="#f39c12", linewidth=2, markersize=8)
ax2.fill_between(t_vals, pct_g, alpha=0.15, color="#f39c12")
ax2.axvline(UMBRAL_CNN, color="red", linestyle="--", linewidth=1.5, label=f"Umbral actual ({UMBRAL_CNN})")
ax2.set_xlabel("Umbral de confianza CNN", fontsize=11)
ax2.set_ylabel("% imágenes procesadas por Gemini", fontsize=11)
ax2.set_title("Coste de API (llamadas a Gemini) vs Umbral", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.suptitle("Tarea 4 — Sensibilidad del umbral de confianza", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("dia3_sensibilidad_umbral.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ dia3_sensibilidad_umbral.png guardada")
'''

RESUMEN = '''\
---

## Resumen del Día 3

### Lo que se evaluó

| Tarea | Tiempo | Resultado |
|-------|--------|-----------|
| Ejecución sobre set de test completo | 1.5h | Agente ejecutado sobre todas las imágenes de test, registrando cada paso |
| Comparativa CNN sola vs Agente completo | 1.5h | Matrices de confusión y métricas side-by-side |
| Análisis de intervenciones de Gemini | 1h | Casos donde corrige, empeora, o coincide con la CNN |
| Análisis de sensibilidad del umbral | 1h | Curvas accuracy/F1 y coste en llamadas API según umbral |

### Ficheros generados

| Fichero | Contenido |
|---------|-----------|
| `dia3_intervencion_gemini.png` | Pie chart quién decide + distribución score MedSAM |
| `dia3_comparativa_confusion.png` | Matrices de confusión: CNN sola vs Agente completo |
| `dia3_comparativa_metricas.png` | Barras comparativas de accuracy, F1, precision, recall |
| `dia3_analisis_gemini.png` | Categorías de intervención de Gemini |
| `dia3_sensibilidad_umbral.png` | Curvas de métricas y coste según umbral |

---

*Documentación del Día 3 — Grupo 5 — Proyecto IA RX — Mayo 2026*
'''

# ── Construir la lista de celdas ──────────────────────────────────────────────
cells = [
    md("# Día 3 — Evaluación del Agente Completo (5h)\n\n"
       "**Grupo 5 — Agente multi-paso (MedSAM + CNN + Gemini)**  \n"
       "**Proyecto**: Clasificación de posicionamiento de dedo en escáner biométrico  \n"
       "**Fecha**: Mayo 2026\n\n"
       "---\n\n"
       "## Objetivo del día\n\n"
       "Ejecutar el agente sobre el set de test completo, comparar su precisión con la CNN sola, "
       "analizar los casos donde Gemini corrige o empeora, y ajustar el umbral de confianza según los resultados.\n\n"
       "| Tarea | Descripción | Tiempo |\n"
       "|-------|-------------|--------|\n"
       "| Tarea 1 | Ejecutar el agente sobre el set de test y registrar % de intervención de Gemini | 1.5h |\n"
       "| Tarea 2 | Comparar precisión del agente multi-paso vs CNN sola | 1.5h |\n"
       "| Tarea 3 | Analizar casos donde Gemini corrige y donde empeora | 1h |\n"
       "| Tarea 4 | Ajustar umbral de confianza según los resultados | 1h |"),

    md("## 0. Imports y configuración\n\nMismos imports que el Día 2 + `sklearn` para métricas de clasificación."),
    code(src_imports),

    md("### Configuración global\n\nMismos parámetros y rutas que el Día 2."),
    code(src_config),

    md("### Cargar MedSAM\n\nCarga robusta del checkpoint ViT-B."),
    code(src_medsam),

    md("### Cargar ResidualCNN\n\nArquitectura idéntica al Día 2 y al entrenamiento original del Grupo 2."),
    code(src_cnn_arch),

    md("### Configurar cliente Gemini (OpenRouter)\n\nAPI key cargada del `.env`. Verificación de conexión antes de continuar."),
    code(src_gemini),

    md("### Clase `AgenteMedico`\n\nMismo código que el Día 2 — se copia aquí para que el notebook sea autónomo."),
    code(src_agente),

    md("### Instanciar el agente"),
    code(src_instancia),

    md("---\n\n## Tarea 1 — Ejecución sobre el set de test completo *(1.5h)*\n\n"
       "Se ejecuta `agente.veredicto_final()` sobre **todas las imágenes del set de test** "
       "(no solo los 10 casos del Día 2). Se registra en cada imagen:\n\n"
       "- `score_medsam`: confianza de la segmentación\n"
       "- `clase_cnn` + `p_cnn`: predicción y confianza de la CNN\n"
       "- `arbitraje_gemini`: si intervino Gemini\n"
       "- `clase_final`: veredicto definitivo\n"
       "- `razon_gemini`: justificación textual (si Gemini intervino)"),
    code(TAREA1_RUN),

    md("### Estadísticas de intervención y distribución de scores"),
    code(TAREA1_STATS),

    md("---\n\n## Tarea 2 — Comparativa: CNN sola vs Agente completo *(1.5h)*\n\n"
       "Se evalúa la CNN **directamente sobre la imagen original** (sin máscara MedSAM, sin Gemini) "
       "para usar como línea base, y se compara con el agente completo.\n\n"
       "Métricas evaluadas: Accuracy, Precision, Recall, F1 por clase y Macro F1."),
    code(TAREA2_CNN),

    md("### Métricas detalladas y matrices de confusión"),
    code(TAREA2_COMPARE),

    md("### Gráfico comparativo de métricas"),
    code(TAREA2_BARS),

    md("---\n\n## Tarea 3 — Análisis de las intervenciones de Gemini *(1h)*\n\n"
       "Se clasifican **todas las imágenes donde Gemini intervino** en cuatro categorías:\n\n"
       "| Categoría | Descripción |\n"
       "|-----------|-------------|\n"
       "| **Gemini corrige** | La CNN se equivocó, Gemini acierta ✅ |\n"
       "| **Gemini empeora** | La CNN acertaba, Gemini se equivoca ❌ |\n"
       "| **Ambos correctos** | CNN y Gemini coinciden y aciertan 🟢 |\n"
       "| **Ambos incorrectos** | CNN y Gemini coinciden y fallan 🔴 |"),
    code(TAREA3_ANALYZE),

    md("### Visualización de los casos de intervención"),
    code(TAREA3_VIS),

    md("---\n\n## Tarea 4 — Ajuste del umbral de confianza *(1h)*\n\n"
       "Se simula qué habría pasado con umbrales distintos al 0.70 original.\n\n"
       "**Metodología**: ya tenemos `p_cnn` y la respuesta de Gemini para todos los casos "
       "donde `p_cnn < 0.70`. Para simular umbrales ≤ 0.70 la simulación es exacta. "
       "Para umbrales > 0.70, los casos nuevos (0.70 ≤ p_cnn < t) no tienen respuesta Gemini "
       "y se usa la CNN como fallback — por eso se marcan como *estimado*.\n\n"
       "El gráfico muestra el trade-off entre calidad (accuracy/F1) y coste (% de llamadas a Gemini)."),
    code(TAREA4_SIM),

    md("### Gráfico de sensibilidad del umbral"),
    code(TAREA4_PLOT),

    md(RESUMEN),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"✓ Notebook generado: {OUT}")
print(f"  Celdas: {len(cells)}")
