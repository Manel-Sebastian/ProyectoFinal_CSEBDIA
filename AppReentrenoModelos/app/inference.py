"""
 inference.py
 ──────────────────────────────────────────────────────────────
 Carga modelos PyTorch entrenados (.pt) y realiza predicciones
 sobre imágenes de radiografías.

 Uso desde app.py:
     from app.inference import predict_image, list_available_models

 Uso directo:
     from app.inference import predict_image
     resultado = predict_image("ruta/a/imagen.jpg", body_part="mano", model_name="resnet18_finetuning")
     print(resultado)  # {"label": "si", "confidence": 0.87}
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights
from PIL import Image

# ------------------------------------------------------------------
# Configuración (puede sobreescribirse con variables de entorno)
# ------------------------------------------------------------------
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./modelos"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASES = ["si", "no"]
LABEL2IDX = {"si": 0, "no": 1}
IDX2LABEL = {0: "si", 1: "no"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# Transformación de evaluación (igual que en entrenamiento)
TRANSFORM_EVAL = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ------------------------------------------------------------------
# Constructores de modelos (DEBEN coincidir con retrain.py)
# ------------------------------------------------------------------
def build_resnet18_finetuning(num_classes: int = 2):
    """ResNet18 con layer3, layer4 y fc descongelados."""
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_feat, num_classes))
    for name, param in model.named_parameters():
        if any(name.startswith(blk) for blk in ["layer3", "layer4", "fc"]):
            param.requires_grad = True
    return model


def build_mobilenet_augmentation(num_classes: int = 2):
    """MobileNetV2 con últimas capas descongeladas."""
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_feat, num_classes))
    for name, param in model.named_parameters():
        if any(x in name for x in ["features.14", "features.15", "features.16",
                                     "features.17", "features.18", "classifier"]):
            param.requires_grad = True
    return model


MODEL_BUILDERS = {
    "resnet18_finetuning": build_resnet18_finetuning,
    "mobilenet_augmentation": build_mobilenet_augmentation,
}


# ------------------------------------------------------------------
# Funciones de carga y predicción
# ------------------------------------------------------------------
def load_model(body_part: str, model_name: str):
    """
    Carga un modelo entrenado desde disco.

    Args:
        body_part:  ej. "mano", "rodilla"
        model_name: ej. "resnet18_finetuning", "mobilenet_augmentation"

    Returns:
        model (nn.Module) listo para evaluación, o None si no existe.
    """
    model_dir = MODELS_DIR / body_part
    weights_path = model_dir / f"{model_name}.pt"

    if not weights_path.exists():
        return None

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Modelo '{model_name}' no está registrado.")

    model = MODEL_BUILDERS[model_name](num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(
    image_path: str,
    body_part: str = "mano",
    model_name: str = "resnet18_finetuning",
) -> Optional[Dict]:
    """
    Predice la clase de una imagen con un modelo local entrenado.

    Args:
        image_path: Ruta a la imagen (jpg/png).
        body_part:  Parte del cuerpo (subcarpeta dentro de modelos/).
        model_name: Nombre del modelo a usar.

    Returns:
        Dict con {"label": "si"|"no", "confidence": float} o None si falla.
    """
    model = load_model(body_part, model_name)
    if model is None:
        return None

    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM_EVAL(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = IDX2LABEL[predicted.item()]
    return {
        "label": label,
        "confidence": round(confidence.item(), 4),
        "model": model_name,
        "body_part": body_part,
    }


def list_available_models(body_part: Optional[str] = None) -> List[Dict]:
    """
    Lista los modelos .pt disponibles en MODELS_DIR.

    Args:
        body_part: Si se indica, filtra por esa parte del cuerpo.

    Returns:
        Lista de dicts con {"body_part", "model_name", "path"}.
    """
    results = []
    search_dir = MODELS_DIR if body_part is None else MODELS_DIR / body_part

    if not search_dir.exists():
        return results

    for pt_file in search_dir.rglob("*.pt"):
        part = pt_file.parent.name if pt_file.parent != MODELS_DIR else ""
        results.append({
            "body_part": part,
            "model_name": pt_file.stem,
            "path": str(pt_file),
        })
    return results


# ------------------------------------------------------------------
# Punto de entrada para pruebas rápidas desde terminal
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Uso: python -m app.inference <imagen.jpg> <body_part> [model_name]")
        print("Ejemplo: python -m app.inference dataset/mano/si/ejemplo.jpg mano resnet18_finetuning")
        sys.exit(1)

    img_path = sys.argv[1]
    part = sys.argv[2]
    mdl = sys.argv[3] if len(sys.argv) > 3 else "resnet18_finetuning"

    print(f"🔍 Prediciendo: {img_path}")
    print(f"   Modelo: {mdl} | Parte: {part}")
    res = predict_image(img_path, part, mdl)
    if res:
        print(f"   Resultado: {res['label'].upper()} (confianza: {res['confidence']:.2%})")
    else:
        print("   ❌ No se encontró el modelo entrenado.")
