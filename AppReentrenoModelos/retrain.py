"""
Módulo de reentrenamiento automático de modelos de visión artificial.
Lógica extraída de MobileNet.ipynb y ResNet18.ipynb.

Estructura esperada del dataset:
    DATASET_DIR/
        <parte_cuerpo>/        (ej: mano, rodilla, codo)
            si/
            no/

Salida:
    MODELS_DIR/
        <parte_cuerpo>/
            resnet18_finetuning.pt
            mobilenet_augmentation.pt
        metrics_history.json   ← historial persistente de métricas
"""

import os
import copy
import time
import json
import warnings
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

# ─── Rutas configurables via variables de entorno ─────────────────────────────
DATASET_DIR = Path(os.getenv("DATASET_DIR", "/home/carlos/dataset/imagenes"))
MODELS_DIR  = Path(os.getenv("MODELS_DIR",  "/home/carlos/modelos"))

# ─── Hiperparámetros ──────────────────────────────────────────────────────────
SEED          = 42
IMG_SIZE      = 224
BATCH_SIZE    = 16
NUM_WORKERS   = 0
TEST_SIZE     = 0.20
VAL_SIZE      = 0.20
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASES        = ["si", "no"]
LABEL2IDX     = {"si": 0, "no": 1}
IDX2LABEL     = {0: "si", 1: "no"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Dataset ──────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, use_albumentations: bool = False):
        self.df                 = df.reset_index(drop=True)
        self.transform          = transform
        self.use_albumentations = use_albumentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["path"]).convert("RGB")
        label = int(row["idx"])
        if self.transform:
            if self.use_albumentations:
                img = self.transform(image=np.array(img))["image"]
            else:
                img = self.transform(img)
        return img, label


# ─── Carga y partición del dataset ────────────────────────────────────────────
def load_dataset(data_dir: Path) -> pd.DataFrame:
    records = []
    for clase in CLASES:
        carpeta = data_dir / clase
        if not carpeta.exists():
            continue
        for img_path in sorted(carpeta.glob("*")):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                records.append({
                    "path":  str(img_path),
                    "label": clase,
                    "idx":   LABEL2IDX[clase],
                })
    if not records:
        raise ValueError(f"No se encontraron imágenes en {data_dir}")
    return pd.DataFrame(records)


def make_splits(df: pd.DataFrame):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED,
        stratify=df["label"], shuffle=True,
    )
    df_tr, df_val = train_test_split(
        df_train, test_size=VAL_SIZE, random_state=SEED,
        stratify=df_train["label"],
    )
    return (
        df_tr.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


# ─── Transforms ───────────────────────────────────────────────────────────────
def _transforms_eval():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _transforms_resnet_train():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _aug_pipeline_albumentations():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.6),
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.6),
        A.GaussNoise(std_range=(0.04, 0.2), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─── DataLoaders ──────────────────────────────────────────────────────────────
def make_dataloaders(df_tr, df_val, df_test, train_transform, use_alb: bool):
    eval_tf = _transforms_eval()
    dl_train = DataLoader(
        ImageDataset(df_tr,   train_transform, use_albumentations=use_alb),
        batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS,
    )
    dl_val = DataLoader(
        ImageDataset(df_val,  eval_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    dl_test = DataLoader(
        ImageDataset(df_test, eval_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    return dl_train, dl_val, dl_test


# ─── Loop de entrenamiento ────────────────────────────────────────────────────
def _run_epoch(model, loader, optimizer, criterion, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss = total_correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            out   = model(imgs)
            loss  = criterion(out, labels)
            preds = out.argmax(dim=1)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss    += loss.item() * imgs.size(0)
        total_correct += (preds == labels).sum().item()
        total         += imgs.size(0)
    return total_loss / total, total_correct / total


def train_model(model, dl_train, dl_val, optimizer, criterion, epochs: int,
                scheduler=None, patience: int = 7, min_delta: float = 1e-4,
                log_fn=None):
    best_wts     = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    es_counter   = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, dl_train, optimizer, criterion, is_train=True)
        vl_loss, vl_acc = _run_epoch(model, dl_val,   optimizer, criterion, is_train=False)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc + min_delta:
            best_val_acc = vl_acc
            best_wts     = copy.deepcopy(model.state_dict())
            es_counter   = 0
        else:
            es_counter += 1

        if scheduler:
            scheduler.step()

        elapsed = time.time() - t0
        msg = (
            f"  [{epoch:3d}/{epochs}]  "
            f"train {tr_acc:.3f}/{tr_loss:.4f}  "
            f"val {vl_acc:.3f}/{vl_loss:.4f}  "
            f"ES:{es_counter}/{patience}  ({elapsed:.1f}s)"
        )
        if log_fn:
            log_fn(msg)

        if es_counter >= patience:
            if log_fn:
                log_fn(f"  ⏹ Early stopping (época {epoch}, mejor val_acc={best_val_acc:.3f})")
            break

    if log_fn:
        log_fn(f"  ✓ Mejor val_acc={best_val_acc:.3f} en {len(history['train_loss'])} épocas")
    model.load_state_dict(best_wts)
    return model, history


# ─── Evaluación ───────────────────────────────────────────────────────────────
def evaluate_model(model, dl_test):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dl_test:
            preds = model(imgs.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    y_true = [IDX2LABEL[l] for l in all_labels]
    y_pred = [IDX2LABEL[p] for p in all_preds]
    return y_true, y_pred


def compute_metrics(model_name: str, body_part: str, y_true, y_pred, history: dict) -> dict:
    kw = dict(zero_division=0)
    return {
        "model":             model_name,
        "body_part":         body_part,
        "timestamp":         datetime.now().isoformat(),
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision_si":      round(precision_score(y_true, y_pred, pos_label="si", **kw), 4),
        "recall_si":         round(recall_score(y_true, y_pred,    pos_label="si", **kw), 4),
        "f1_si":             round(f1_score(y_true, y_pred,        pos_label="si", **kw), 4),
        "precision_no":      round(precision_score(y_true, y_pred, pos_label="no", **kw), 4),
        "recall_no":         round(recall_score(y_true, y_pred,    pos_label="no", **kw), 4),
        "f1_no":             round(f1_score(y_true, y_pred,        pos_label="no", **kw), 4),
        "val_loss_best":     round(min(history["val_loss"]),   4) if history["val_loss"]   else None,
        "train_loss_best":   round(min(history["train_loss"]), 4) if history["train_loss"] else None,
        "epochs_trained":    len(history["train_loss"]),
        "test_samples":      len(y_true),
        "device":            str(device),
    }


# ─── Constructores de modelos ──────────────────────────────────────────────────
def build_resnet18_finetuning(num_classes: int = 2):
    """ResNet18 con layer3, layer4 y fc descongelados (fine-tuning parcial)."""
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_feat  = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_feat, num_classes))
    for name, param in model.named_parameters():
        if any(name.startswith(blk) for blk in ["layer3", "layer4", "fc"]):
            param.requires_grad = True
    return model.to(device)


def build_mobilenet_augmentation(num_classes: int = 2):
    """MobileNetV2 con últimas capas del backbone y classifier descongelados."""
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_feat          = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(in_feat, num_classes))
    for name, param in model.named_parameters():
        if any(x in name for x in
               ["features.14", "features.15", "features.16",
                "features.17", "features.18", "classifier"]):
            param.requires_grad = True
    return model.to(device)


# ─── Registro de modelos ──────────────────────────────────────────────────────
# Para añadir un nuevo modelo: añade una entrada con build_fn y parámetros.
MODEL_REGISTRY: dict = {
    "resnet18_finetuning": {
        "build_fn":         build_resnet18_finetuning,
        "epochs":           20,
        "lr_backbone":      1e-6,
        "lr_head":          1e-5,
        "use_augmentation": False,
        "head_prefixes":    ("fc",),
        "filename":         "resnet18_finetuning.pt",
        "description":      "ResNet18 fine-tuning (layer3+layer4+fc)",
    },
    "mobilenet_augmentation": {
        "build_fn":         build_mobilenet_augmentation,
        "epochs":           15,
        "lr_backbone":      1e-5,
        "lr_head":          1e-3,
        "use_augmentation": True,
        "head_prefixes":    ("classifier",),
        "filename":         "mobilenet_augmentation.pt",
        "description":      "MobileNetV2 fine-tuning con data augmentation",
    },
}


# ─── Entrenamiento de un modelo concreto ──────────────────────────────────────
def train_one_model(model_key: str, df_tr, df_val, df_test,
                    body_part: str, output_dir: Path, log_fn=None) -> dict:
    cfg = MODEL_REGISTRY[model_key]

    if cfg["use_augmentation"]:
        train_tf = _aug_pipeline_albumentations()
        use_alb  = True
    else:
        train_tf = _transforms_resnet_train()
        use_alb  = False

    dl_train, dl_val, dl_test = make_dataloaders(df_tr, df_val, df_test, train_tf, use_alb)

    model     = cfg["build_fn"]()
    criterion = nn.CrossEntropyLoss()

    head_prefixes   = cfg["head_prefixes"]
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(n.startswith(h) for h in head_prefixes)]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and any(n.startswith(h) for h in head_prefixes)]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg["lr_backbone"]})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg["lr_head"]})

    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    model, history = train_model(
        model, dl_train, dl_val, optimizer, criterion,
        epochs=cfg["epochs"], scheduler=scheduler, log_fn=log_fn,
    )

    y_true, y_pred = evaluate_model(model, dl_test)
    metrics     = compute_metrics(model_key, body_part, y_true, y_pred, history)
    new_score   = metrics[BEST_METRIC_KEY]
    models_dir  = output_dir.parent
    best_score  = _get_best_score(body_part, model_key, models_dir)
    is_best     = new_score > best_score

    metrics["saved_as_best"] = is_best
    metrics[f"best_{BEST_METRIC_KEY}_prev"] = round(best_score, 4) if best_score >= 0 else None

    if log_fn:
        log_fn(
            f"  📊 Accuracy={metrics['accuracy']:.3f}  "
            f"F1(si)={new_score:.3f}  "
            f"Recall(si)={metrics['recall_si']:.3f}"
        )

    if is_best:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / cfg["filename"]
        torch.save(model.state_dict(), model_path)
        prev_str = f"{best_score:.3f}" if best_score >= 0 else "ninguno"
        if log_fn:
            log_fn(f"  🏆 Mejor modelo guardado  F1={new_score:.3f} > anterior={prev_str}")
            log_fn(f"  💾 {model_path}")
    else:
        if log_fn:
            log_fn(
                f"  ⏭  No se sobreescribe — F1={new_score:.3f} ≤ "
                f"mejor registrado={best_score:.3f}"
            )

    return metrics


# ─── Mejor score histórico ────────────────────────────────────────────────────
BEST_METRIC_KEY = "f1_si"   # métrica usada para decidir qué modelo es "mejor"

def _get_best_score(body_part: str, model_key: str, models_dir: Path) -> float:
    """Devuelve el mejor score registrado para esta combinación (parte, modelo).
    Retorna -1 si no hay historial previo."""
    history = load_metrics_history(models_dir)
    scores  = [
        m[BEST_METRIC_KEY] for m in history
        if m.get("body_part") == body_part
        and m.get("model") == model_key
        and m.get("saved_as_best", False)
    ]
    return max(scores) if scores else -1.0


# ─── Persistencia de métricas ─────────────────────────────────────────────────
def _save_metrics(new_metrics: list, models_dir: Path) -> None:
    metrics_file = models_dir / "metrics_history.json"
    history = []
    if metrics_file.exists():
        with open(metrics_file, encoding="utf-8") as f:
            history = json.load(f)
    history.extend(new_metrics)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def load_metrics_history(models_dir: Path = None) -> list:
    """Devuelve el historial completo de métricas desde disco."""
    models_dir   = Path(models_dir) if models_dir else MODELS_DIR
    metrics_file = models_dir / "metrics_history.json"
    if not metrics_file.exists():
        return []
    with open(metrics_file, encoding="utf-8") as f:
        return json.load(f)


# ─── Función principal ────────────────────────────────────────────────────────
def retrain_all_models(
    dataset_dir=None,
    models_dir=None,
    log_fn=None,
    progress_fn=None,
    model_keys: list = None,
) -> list:
    """
    Reentrena todos los modelos registrados para cada parte del cuerpo
    encontrada en dataset_dir.

    Args:
        dataset_dir:  Ruta al directorio raíz del dataset. Por defecto DATASET_DIR.
        models_dir:   Ruta donde se guardan los modelos. Por defecto MODELS_DIR.
        log_fn:       Callable(str) para mensajes en tiempo real (ej: print).
        progress_fn:  Callable(float) con valor 0.0–1.0 indicando progreso global.

    Returns:
        Lista de dicts con métricas de cada modelo entrenado en esta ejecución.
    """
    dataset_dir = Path(dataset_dir) if dataset_dir else DATASET_DIR
    models_dir  = Path(models_dir)  if models_dir  else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    log(f"🖥  Dispositivo : {device}")
    log(f"📂 Dataset     : {dataset_dir}")
    log(f"💾 Modelos     : {models_dir}")

    if not dataset_dir.exists():
        log(f"❌ ERROR: El directorio del dataset no existe: {dataset_dir}")
        return []

    # Caso 1: la carpeta apuntada tiene si/ y no/ directamente
    if (dataset_dir / "si").exists() and (dataset_dir / "no").exists():
        body_parts   = [dataset_dir.name]
        dataset_dir  = dataset_dir.parent

    # Caso 2: la carpeta contiene subcarpetas <parte>/si/ y <parte>/no/
    else:
        body_parts = sorted([
            d.name for d in dataset_dir.iterdir()
            if d.is_dir() and (d / "si").exists() and (d / "no").exists()
        ])

    if not body_parts:
        log(f"❌ ERROR: No se encontraron imágenes en {dataset_dir}")
        log("   La carpeta debe contener si/ y no/ directamente,")
        log("   o subcarpetas con estructura <parte>/si/ y <parte>/no/")
        return []

    modelos_a_entrenar = {
        k: v for k, v in MODEL_REGISTRY.items()
        if model_keys is None or k in model_keys
    }

    log(f"\n📋 Partes del cuerpo : {body_parts}")
    log(f"🤖 Modelos           : {list(modelos_a_entrenar.keys())}")
    total_tasks = len(body_parts) * len(modelos_a_entrenar)
    log(f"📝 Total tareas      : {total_tasks}\n")

    completed   = 0
    all_metrics = []

    for body_part in body_parts:
        data_dir = dataset_dir / body_part
        log(f"\n{'='*60}")
        log(f"  📍 Parte del cuerpo: {body_part.upper()}")
        log(f"{'='*60}")

        try:
            df = load_dataset(data_dir)
            df_tr, df_val, df_test = make_splits(df)
            log(f"  Imágenes: {len(df)} total  "
                f"(train={len(df_tr)}, val={len(df_val)}, test={len(df_test)})")
            log(f"  Distribución: {df['label'].value_counts().to_dict()}")
        except Exception as exc:
            log(f"  ❌ Error cargando dataset: {exc}")
            completed += len(MODEL_REGISTRY)
            if progress_fn:
                progress_fn(completed / total_tasks)
            continue

        output_dir = models_dir / body_part

        for model_key, cfg in modelos_a_entrenar.items():
            log(f"\n  ▶ {model_key} — {cfg['description']}")
            try:
                metrics = train_one_model(
                    model_key, df_tr, df_val, df_test,
                    body_part, output_dir, log_fn=log,
                )
                all_metrics.append(metrics)
            except Exception as exc:
                log(f"  ❌ Error entrenando {model_key}: {exc}")
                log(traceback.format_exc())

            completed += 1
            if progress_fn:
                progress_fn(completed / total_tasks)

    if all_metrics:
        _save_metrics(all_metrics, models_dir)
        log(f"\n✅ Completado: {len(all_metrics)} modelos entrenados y guardados.")
        log(f"📄 Historial: {models_dir / 'metrics_history.json'}")
    else:
        log("\n⚠️  No se entrenó ningún modelo.")

    return all_metrics


if __name__ == "__main__":
    retrain_all_models()
