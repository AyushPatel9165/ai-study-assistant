import os
import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.classification.dataset import DataPipeline
from src.classification.model import QuestionClassifierMLP

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward propagation
        logits = model(features)
        loss = criterion(logits, targets)

        # Backpropagation
        loss.backward()

        # Weight updates
        optimizer.step()

        # Tracking metrics
        running_loss += loss.item() * features.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc

def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = criterion(logits, targets)

            running_loss += loss.item() * features.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc, np.array(all_targets), np.array(all_preds)

def train_classifier(
    data_csv: str = "data/raw/student_questions.csv",
    models_dir: str = "models",
    epochs: int = 35,
    lr: float = 0.001,
    batch_size: int = 16,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")

    # 1. Prepare data pipeline
    pipeline = DataPipeline(max_features=800, batch_size=batch_size, random_state=42)
    prep_data = pipeline.load_and_prepare(data_csv)

    train_loader = prep_data["train_loader"]
    val_loader = prep_data["val_loader"]
    test_loader = prep_data["test_loader"]
    vectorizer = prep_data["vectorizer"]
    label_encoder = prep_data["label_encoder"]
    classes = prep_data["classes"]
    feature_dim = prep_data["feature_dim"]
    num_classes = prep_data["num_classes"]

    print(f"[INFO] Dataset loaded: {len(classes)} classes -> {classes}")
    print(f"[INFO] Vocabulary / feature dimension: {feature_dim}")

    # 2. Instantiate Model, Loss, Optimizer
    model = QuestionClassifierMLP(
        input_dim=feature_dim,
        hidden_dim1=128,
        hidden_dim2=64,
        num_classes=num_classes,
        dropout_rate=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_weights = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"[INFO] Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%"
            )

    # Load best weights for testing
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 3. Comprehensive Evaluation on Test Set
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    conf_mat = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION RESULTS:")
    print("=" * 60)
    print(f"Test Accuracy : {test_acc*100:.2f}%")
    print(f"Weighted F1   : {f1:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", conf_mat)
    print("=" * 60)

    # 4. Save Model Artifacts
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "question_classifier.pt")
    vec_save_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    encoder_save_path = os.path.join(models_dir, "label_encoder.pkl")
    meta_save_path = os.path.join(models_dir, "model_metadata.json")

    # Save PyTorch model state
    torch.save(model.state_dict(), model_save_path)
    joblib.dump(vectorizer, vec_save_path)
    joblib.dump(label_encoder, encoder_save_path)

    metadata = {
        "input_dim": feature_dim,
        "classes": classes,
        "test_accuracy": float(test_acc),
        "test_f1_weighted": float(f1),
        "test_f1_macro": float(macro_f1),
        "test_loss": float(test_loss),
        "confusion_matrix": conf_mat.tolist(),
        "epochs_trained": epochs,
        "hidden_dim1": 128,
        "hidden_dim2": 64
    }

    with open(meta_save_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SUCCESS] Model saved to {model_save_path}")
    print(f"[SUCCESS] Vectorizer saved to {vec_save_path}")
    print(f"[SUCCESS] Label encoder saved to {encoder_save_path}")
    print(f"[SUCCESS] Metadata saved to {meta_save_path}")

    return {
        "test_accuracy": test_acc,
        "test_f1": f1,
        "confusion_matrix": conf_mat,
        "classification_report": report,
        "history": history,
        "metadata": metadata
    }

if __name__ == "__main__":
    train_classifier()
