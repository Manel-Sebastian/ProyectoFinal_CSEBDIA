"""Genera demo_agente_dia4.ipynb — notebook de demostración del Día 4."""
import json
from pathlib import Path

DIA2 = Path(__file__).parent.parent / "Dia2" / "agente_medico_dia2.ipynb"
OUT  = Path(__file__).parent / "demo_agente_dia4.ipynb"

with open(DIA2) as f:
    nb2 = json.load(f)

def code(src): return {"cell_type":"code","metadata":{},"source":[src],"outputs":[],"execution_count":None}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":[src]}

src_imports   = "".join(nb2["cells"][2]["source"])
src_config    = "".join(nb2["cells"][4]["source"])
src_medsam    = "".join(nb2["cells"][6]["source"])
src_cnn_arch  = "".join(nb2["cells"][8]["source"])
src_gemini    = "".join(nb2["cells"][10]["source"])
src_agente    = "".join(nb2["cells"][12]["source"])
src_instancia = "".join(nb2["cells"][14]["source"])

# ── Código de la demo ─────────────────────────────────────────────────────────

SELECCION = '''\
import random, os
random.seed(7)

test_si = sorted((DATASET_DIR / "test" / "si").glob("*.png"))
test_no = sorted((DATASET_DIR / "test" / "no").glob("*.png"))
all_imgs = [(p, "SI") for p in test_si] + [(p, "NO") for p in test_no]

# Preseleccionar candidatos por confianza CNN
model_cnn.eval()
candidatos = []
for img_path, clase_real in all_imgs:
    tensor = CNN_TRANSFORM(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model_cnn(tensor), dim=1).squeeze().cpu().numpy()
    candidatos.append({"path": img_path, "clase_real": clase_real, "p_cnn": float(probs.max())})

candidatos.sort(key=lambda x: x["p_cnn"], reverse=True)
demo_alta    = candidatos[0]   # mayor confianza CNN → CNN decide
demo_baja    = candidatos[-1]  # menor confianza CNN → Gemini arbitra

print(f"Demo 1 (CNN decide):    {demo_alta['path'].name}  — p_cnn = {demo_alta['p_cnn']:.4f}  — real: {demo_alta['clase_real']}")
print(f"Demo 2 (Gemini arbitra): {demo_baja['path'].name}  — p_cnn = {demo_baja['p_cnn']:.4f}  — real: {demo_baja['clase_real']}")
'''

DEMO1_VIS = '''\
img_path = demo_alta["path"]
img      = np.array(Image.open(img_path).convert("RGB"))
clase_real = demo_alta["clase_real"]

# ── Paso 1: MedSAM ───────────────────────────────────────────────────────────
mask, score_sam = agente.segmentar(img)
sam_overlay     = img.copy()
sam_overlay[mask] = (0.45 * sam_overlay[mask] + 0.55 * np.array([50, 220, 80])).astype(np.uint8)

# ── Paso 2: CNN ───────────────────────────────────────────────────────────────
img_masked = img.copy(); img_masked[~mask] = 0
clase_cnn, p_cnn = agente.clasificar(img, mask)

tensor_vis = CNN_TRANSFORM(Image.fromarray(img_masked.astype(np.uint8))).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    probs_all = torch.softmax(model_cnn(tensor_vis), dim=1).squeeze().cpu().numpy()

# ── Layout: 5 columnas ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle(f"DEMO 1 — CNN decide directamente  |  Imagen real: {clase_real}",
             fontsize=14, fontweight="bold", color="#2980b9")

titles = ["① Imagen original", "② MedSAM — Mascara", "③ CNN input (fondo negro)",
          "④ CNN — Probabilidades", "⑤ VEREDICTO FINAL"]

# ① Original
axes[0].imshow(img)
axes[0].set_title(titles[0], fontsize=10, fontweight="bold")
axes[0].axis("off")

# ② MedSAM overlay
axes[1].imshow(sam_overlay)
color_s = "#27ae60" if score_sam >= UMBRAL_MEDSAM else "#e74c3c"
axes[1].set_title(titles[1], fontsize=10, fontweight="bold")
axes[1].set_xlabel(f"Score MedSAM: {score_sam:.4f}", fontsize=10, color=color_s)
axes[1].axis("off")

# ③ Imagen enmascarada
axes[2].imshow(img_masked)
axes[2].set_title(titles[2], fontsize=10, fontweight="bold")
axes[2].axis("off")

# ④ Barras de probabilidad CNN
clases = ["NO", "SI"]
colores_barra = ["#e74c3c" if c == "NO" else "#27ae60" for c in clases]
bars = axes[3].bar(clases, probs_all[[0,1]], color=colores_barra, alpha=0.85, width=0.5)
for bar, val in zip(bars, probs_all[[0,1]]):
    axes[3].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.1%}", ha="center", fontsize=13, fontweight="bold")
axes[3].axhline(UMBRAL_CNN, color="gray", linestyle="--", linewidth=1.2,
                label=f"Umbral {UMBRAL_CNN}")
axes[3].set_ylim(0, 1.15); axes[3].set_ylabel("Probabilidad")
axes[3].set_title(titles[3], fontsize=10, fontweight="bold")
axes[3].legend(fontsize=8); axes[3].grid(axis="y", alpha=0.3)

# ⑤ Veredicto final
acierto   = clase_cnn == clase_real
color_v   = "#27ae60" if acierto else "#e74c3c"
simbolo   = "✓" if acierto else "✗"
axes[4].set_facecolor(color_v + "22")
axes[4].text(0.5, 0.70, clase_cnn,    ha="center", va="center", fontsize=40,
             fontweight="bold", color=color_v, transform=axes[4].transAxes)
axes[4].text(0.5, 0.42, f"CNN directa  |  p_cnn = {p_cnn:.1%}", ha="center", va="center",
             fontsize=11, color="#555", transform=axes[4].transAxes)
axes[4].text(0.5, 0.18, f"{simbolo} real={clase_real}", ha="center", va="center",
             fontsize=12, fontweight="bold", color=color_v, transform=axes[4].transAxes)
for spine in axes[4].spines.values():
    spine.set_edgecolor(color_v); spine.set_linewidth(3)
axes[4].set_title(titles[4], fontsize=10, fontweight="bold")
axes[4].set_xticks([]); axes[4].set_yticks([])

plt.tight_layout()
plt.savefig("dia4_demo1_cnn_directa.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✓ DEMO 1 completada — CNN decide: {clase_cnn} ({p_cnn:.1%})")
'''

DEMO2_MEDSAM = '''\
img_path_2 = demo_baja["path"]
img2       = np.array(Image.open(img_path_2).convert("RGB"))
clase_real2 = demo_baja["clase_real"]

# ── Paso 1: MedSAM ───────────────────────────────────────────────────────────
mask2, score_sam2 = agente.segmentar(img2)
sam_overlay2 = img2.copy()
sam_overlay2[mask2] = (0.45 * sam_overlay2[mask2] + 0.55 * np.array([50, 220, 80])).astype(np.uint8)

img_masked2 = img2.copy(); img_masked2[~mask2] = 0

# ── Paso 2: CNN ───────────────────────────────────────────────────────────────
clase_cnn2, p_cnn2 = agente.clasificar(img2, mask2)

tensor_vis2 = CNN_TRANSFORM(Image.fromarray(img_masked2.astype(np.uint8))).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    probs_all2 = torch.softmax(model_cnn(tensor_vis2), dim=1).squeeze().cpu().numpy()

# ── Layout: 3 primeras columnas (mismo que demo 1) ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle(f"DEMO 2 — Paso 1+2: MedSAM + CNN incierta  |  Imagen real: {clase_real2}",
             fontsize=13, fontweight="bold", color="#e67e22")

axes[0].imshow(img2)
axes[0].set_title("① Imagen original", fontsize=10, fontweight="bold"); axes[0].axis("off")

color_s2 = "#27ae60" if score_sam2 >= UMBRAL_MEDSAM else "#e74c3c"
axes[1].imshow(sam_overlay2)
axes[1].set_title("② MedSAM — Máscara", fontsize=10, fontweight="bold")
axes[1].set_xlabel(f"Score MedSAM: {score_sam2:.4f}", fontsize=10, color=color_s2)
axes[1].axis("off")

clases = ["NO", "SI"]
colores_b = ["#e74c3c" if c == "NO" else "#27ae60" for c in clases]
bars2 = axes[2].bar(clases, probs_all2[[0,1]], color=colores_b, alpha=0.85, width=0.5)
for bar, val in zip(bars2, probs_all2[[0,1]]):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.1%}", ha="center", fontsize=13, fontweight="bold")
axes[2].axhline(UMBRAL_CNN, color="gray", linestyle="--", linewidth=1.5,
                label=f"Umbral CNN ({UMBRAL_CNN})")
axes[2].set_ylim(0, 1.15); axes[2].set_ylabel("Probabilidad")
axes[2].set_title(f"CNN INCIERTA  p_cnn={p_cnn2:.1%} < {UMBRAL_CNN}  -> Gemini arbitra",
                  fontsize=10, fontweight="bold", color="#e67e22")
axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("dia4_demo2_paso1y2.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"CNN incierta: {clase_cnn2} con {p_cnn2:.1%} → Se activa Gemini como árbitro")
'''

DEMO2_GEMINI = '''\
# ── Paso 3: Gemini arbitra ────────────────────────────────────────────────────
print("Consultando a Gemini como árbitro...")
clase_final2, razon2 = agente.consultar_gemini(img2, mask2, clase_cnn2, p_cnn2)
print(f"  Gemini responde: {clase_final2}")
print(f"  Razón: {razon2}")

acierto2  = clase_final2 == clase_real2
color_v2  = "#27ae60" if acierto2 else "#e74c3c"
simbolo2  = "✓" if acierto2 else "✗"

# ── Layout: overlay Gemini + veredicto ───────────────────────────────────────
overlay2 = img2.copy()
overlay2[mask2] = (0.5 * overlay2[mask2] + 0.5 * np.array([50, 220, 80])).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(f"DEMO 2 — Paso 3: Gemini arbitra  |  Imagen real: {clase_real2}",
             fontsize=13, fontweight="bold", color="#16a085")

# Imagen 1 enviada a Gemini
axes[0].imshow(img2)
axes[0].set_title("IMAGEN 1 enviada a Gemini\\nFotografia original", fontsize=10, fontweight="bold")
axes[0].axis("off")

# Imagen 2 enviada a Gemini
axes[1].imshow(overlay2)
axes[1].set_title("IMAGEN 2 enviada a Gemini\\nMascara MedSAM (verde)", fontsize=10, fontweight="bold")
axes[1].axis("off")

# Veredicto de Gemini
axes[2].set_facecolor(color_v2 + "22")
axes[2].text(0.5, 0.72, clase_final2, ha="center", va="center", fontsize=40,
             fontweight="bold", color=color_v2, transform=axes[2].transAxes)
axes[2].text(0.5, 0.50, "Gemini árbitro", ha="center", va="center",
             fontsize=10, color="#555", transform=axes[2].transAxes)

razon_wrap = razon2[:60] + ("..." if len(razon2) > 60 else "")
axes[2].text(0.5, 0.34, f'"{razon_wrap}"', ha="center", va="center",
             fontsize=8, color="#333", style="italic", wrap=True,
             transform=axes[2].transAxes)
axes[2].text(0.5, 0.12, f"{simbolo2} real={clase_real2} CNN={clase_cnn2}",
             ha="center", va="center", fontsize=11, fontweight="bold",
             color=color_v2, transform=axes[2].transAxes)
for spine in axes[2].spines.values():
    spine.set_edgecolor(color_v2); spine.set_linewidth(3)
axes[2].set_title("VEREDICTO FINAL (Gemini arbitro)", fontsize=10, fontweight="bold")
axes[2].set_xticks([]); axes[2].set_yticks([])

plt.tight_layout()
plt.savefig("dia4_demo2_gemini.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\\n✓ DEMO 2 completada — Gemini decide: {clase_final2}")
'''

RESUMEN_VIS = '''\
# Cuadro comparativo de las dos demos
fig, axes = plt.subplots(2, 5, figsize=(24, 9))
fig.suptitle("Pipeline completo — Las dos demos lado a lado", fontsize=15, fontweight="bold")

col_titles = ["① Imagen original", "② MedSAM\\n(mascara)", "③ CNN input\\n(enmascarado)",
              "④ CNN\\nprobabilidades", "⑤ Veredicto\\nfinal"]
for j, t in enumerate(col_titles):
    axes[0, j].set_title(t, fontsize=9, fontweight="bold", pad=6)

def fill_row(row, img, mask, img_masked, probs, clase_cnn, p_cnn, clase_final,
             clase_real, via_gemini, razon=None):
    color_row = "#2980b9" if not via_gemini else "#16a085"
    acierto   = clase_final == clase_real
    color_v   = "#27ae60" if acierto else "#e74c3c"

    ov = img.copy()
    ov[mask] = (0.45 * ov[mask] + 0.55 * np.array([50, 220, 80])).astype(np.uint8)

    axes[row, 0].imshow(img)
    _lbl = "Gemini arbitra" if via_gemini else "CNN directa"
    axes[row, 0].set_ylabel(
        f"Real: {clase_real}\\n{_lbl}",
        fontsize=9, color=color_row, fontweight="bold"
    )
    axes[row, 0].axis("off")

    axes[row, 1].imshow(ov); axes[row, 1].axis("off")

    axes[row, 2].imshow(img_masked); axes[row, 2].axis("off")

    clases = ["NO", "SI"]
    colores_b = ["#e74c3c", "#27ae60"]
    axes[row, 3].bar(clases, probs[[0,1]], color=colores_b, alpha=0.85, width=0.5)
    axes[row, 3].axhline(UMBRAL_CNN, color="gray", linestyle="--", linewidth=1)
    axes[row, 3].set_ylim(0, 1.15)
    axes[row, 3].set_title(f"CNN: {clase_cnn} ({p_cnn:.1%})", fontsize=8,
                            color="#e67e22" if via_gemini else "#27ae60")
    axes[row, 3].grid(axis="y", alpha=0.3)

    axes[row, 4].set_facecolor(color_v + "22")
    axes[row, 4].text(0.5, 0.65, clase_final, ha="center", va="center",
                      fontsize=36, fontweight="bold", color=color_v,
                      transform=axes[row, 4].transAxes)
    via_txt = "vía Gemini" if via_gemini else "vía CNN"
    axes[row, 4].text(0.5, 0.38, via_txt, ha="center", va="center",
                      fontsize=9, color="#555", transform=axes[row, 4].transAxes)
    simbolo = "✓ CORRECTO" if acierto else "✗ INCORRECTO"
    axes[row, 4].text(0.5, 0.18, simbolo, ha="center", va="center",
                      fontsize=11, fontweight="bold", color=color_v,
                      transform=axes[row, 4].transAxes)
    for spine in axes[row, 4].spines.values():
        spine.set_edgecolor(color_v); spine.set_linewidth(3)
    axes[row, 4].set_xticks([]); axes[row, 4].set_yticks([])

fill_row(0, img, mask, img_masked, probs_all, clase_cnn, p_cnn,
         clase_cnn, clase_real, via_gemini=False)
fill_row(1, img2, mask2, img_masked2, probs_all2, clase_cnn2, p_cnn2,
         clase_final2, clase_real2, via_gemini=True, razon=razon2)

plt.tight_layout()
plt.savefig("dia4_pipeline_completo.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ dia4_pipeline_completo.png guardada")
'''

DIAGRAMA = '''\
# Diagrama de flujo en texto — reproducible sin Mermaid
print("""
╔══════════════════════════════════════════════════════════════════╗
║           PIPELINE AGENTE MEDICO — GRUPO 5                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   [Imagen PNG 640×480]                                          ║
║          │                                                       ║
║          ▼                                                       ║
║   ┌─────────────────────────────────────┐                       ║
║   │ PASO 1 — MedSAM (ViT-B)            │                       ║
║   │  Entrada: imagen RGB                │                       ║
║   │  Salida : máscara binaria + score   │                       ║
║   └─────────────────────────────────────┘                       ║
║          │                                                       ║
║    score >= 0.60?                                               ║
║      No  │  Sí                                                  ║
║     ─────┤                                                       ║
║    ERROR │                                                       ║
║          ▼                                                       ║
║   ┌─────────────────────────────────────┐                       ║
║   │ PASO 2 — ResidualCNN (2.8M params) │                       ║
║   │  Entrada: imagen con fondo negro    │                       ║
║   │  Salida : clase (SI/NO) + p_cnn    │                       ║
║   └─────────────────────────────────────┘                       ║
║          │                                                       ║
║    p_cnn >= 0.70?                                               ║
║      Sí  │  No                                                  ║
║    ──────┤  ────────────────────────────┐                       ║
║          │                              ▼                        ║
║          │          ┌──────────────────────────────────┐        ║
║          │          │ PASO 3 — Gemini 2.0 Flash Lite  │        ║
║          │          │  Entrada: img + overlay + CNN    │        ║
║          │          │  Salida : clase + razón texto    │        ║
║          │          └──────────────────────────────────┘        ║
║          │                              │                        ║
║          ▼                              ▼                        ║
║   [VEREDICTO CNN]              [VEREDICTO GEMINI]               ║
║   Alta confianza               + RAZÓN en texto                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

print(f"Umbrales:")
print(f"  UMBRAL_MEDSAM = {UMBRAL_MEDSAM}  → segmentación mínima aceptable")
print(f"  UMBRAL_CNN    = {UMBRAL_CNN}  → por debajo, Gemini arbitra")
print(f"\\nModelos:")
print(f"  MedSAM   : ViT-B, ~90M params — segment_anything")
print(f"  CNN      : ResidualCNN, 2.79M params — entrenada por Grupo 2")
print(f"  Gemini   : gemini-2.0-flash-lite-001 vía OpenRouter (zero-shot)")
'''

cells = [
    md("# Día 4 — Notebook de Demostración del Agente Multi-Paso\n\n"
       "**Grupo 5 — Agente multi-paso (MedSAM + CNN + Gemini)**  \n"
       "**Proyecto**: Clasificación de posicionamiento de dedo en escáner biométrico  \n"
       "**Fecha**: Mayo 2026\n\n"
       "---\n\n"
       "## Objetivo\n\n"
       "Este notebook es la **demostración presentable** del agente. Muestra el pipeline completo "
       "paso a paso sobre dos imágenes reales:\n\n"
       "- **Demo 1**: imagen donde la CNN tiene alta confianza → el agente decide sin Gemini\n"
       "- **Demo 2**: imagen donde la CNN es incierta → Gemini actúa como árbitro\n\n"
       "Cada paso del pipeline se visualiza con la imagen intermedia resultante.\n\n"
       "---"),

    md("## 0. Carga de modelos\n\nMisma configuración que los notebooks de Día 2 y Día 3."),
    code(src_imports),
    code(src_config),
    code(src_medsam),
    code(src_cnn_arch),
    code(src_gemini),
    code(src_agente),
    code(src_instancia),

    md("---\n\n## Diagrama de flujo del agente\n\n"
       "Representación del pipeline completo antes de las demostraciones."),
    code(DIAGRAMA),

    md("---\n\n## Selección de imágenes de demostración\n\n"
       "Se busca automáticamente:\n"
       "- La imagen con **mayor** `p_cnn` → CNN muy segura (Demo 1)\n"
       "- La imagen con **menor** `p_cnn` → CNN muy insegura (Demo 2)"),
    code(SELECCION),

    md("---\n\n## Demo 1 — CNN decide directamente (`p_cnn ≥ 0.70`)\n\n"
       "El agente ejecuta MedSAM (Paso 1) y la CNN (Paso 2). Como la confianza es alta, "
       "el veredicto es directamente el de la CNN — Gemini no interviene.\n\n"
       "Se muestran los 5 pasos del pipeline en una sola fila:\n\n"
       "| ① | ② | ③ | ④ | ⑤ |\n"
       "|---|---|---|---|---|\n"
       "| Imagen original | Máscara MedSAM | CNN input (fondo negro) | Probabilidades CNN | Veredicto final |"),
    code(DEMO1_VIS),

    md("---\n\n## Demo 2 — Gemini arbitra (`p_cnn < 0.70`)\n\n"
       "### Pasos 1 y 2: MedSAM + CNN incierta\n\n"
       "La CNN produce una confianza baja — el agente detecta que no es suficientemente seguro "
       "y activa el árbitro Gemini."),
    code(DEMO2_MEDSAM),

    md("### Paso 3: Gemini recibe dos imágenes y emite veredicto\n\n"
       "Gemini recibe:\n"
       "- **Imagen 1**: la fotografía original del dedo sobre el escáner\n"
       "- **Imagen 2**: la misma imagen con la máscara MedSAM coloreada en verde\n\n"
       "Gemini devuelve un veredicto (`SI`/`NO`) y una razón en texto."),
    code(DEMO2_GEMINI),

    md("---\n\n## Pipeline completo — Ambas demos lado a lado\n\n"
       "Visualización comparativa de las dos demos en el mismo grid:\n\n"
       "- **Fila 1** (azul): CNN decide directamente\n"
       "- **Fila 2** (verde oscuro): Gemini actúa como árbitro"),
    code(RESUMEN_VIS),

    md("---\n\n## Resumen\n\n"
       "| Demo | Modelo que decide | Confianza CNN | Veredicto | Correcto |\n"
       "|------|------------------|---------------|-----------|----------|\n"
       "| Demo 1 | CNN directa | Alta (≥ 0.70) | clase_cnn | ver resultado |\n"
       "| Demo 2 | Gemini árbitro | Baja (< 0.70) | clase_gemini + razón | ver resultado |\n\n"
       "### Cuándo interviene Gemini\n\n"
       "Gemini **no interviene siempre** — solo en los casos donde la CNN duda. "
       "Esto reduce el coste de API y la latencia en la mayoría de imágenes. "
       "El umbral 0.70 se eligió porque Gemini (70% accuracy zero-shot) es mejor árbitro "
       "que la CNN (60% accuracy) en los casos inciertos.\n\n"
       "---\n\n"
       "*Demostración del Día 4 — Grupo 5 — Proyecto IA RX — Mayo 2026*"),
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"✓ Demo notebook generado: {OUT}")
print(f"  Celdas: {len(cells)}")
