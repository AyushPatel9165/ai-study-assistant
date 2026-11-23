import torch
import torch.nn as nn
from typing import Dict, Any

class QuestionClassifierMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for student question classification into Python, AI, and ML.

    Architecture Overview:
    1. Input Layer:
       Receives TF-IDF representation vector of length `input_dim`.
    2. Hidden Layer 1:
       Linear transformation (input_dim -> hidden_dim1) with trainable weights W1 and bias b1.
       Batch Normalization stabilizes internal covariate shifts.
       ReLU activation introduces non-linearity: max(0, x).
       Dropout randomly zeroes nodes during training to prevent co-adaptation/overfitting.
    3. Hidden Layer 2:
       Linear transformation (hidden_dim1 -> hidden_dim2) with weights W2 and bias b2.
       Batch Normalization + ReLU + Dropout.
    4. Output Layer:
       Linear transformation (hidden_dim2 -> num_classes) producing raw logits.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        num_classes: int = 3,
        dropout_rate: float = 0.3
    ):
        super(QuestionClassifierMLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Layer 1: Input -> Hidden 1
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Layer 2: Hidden 1 -> Hidden 2
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate * 0.7)

        # Output Layer: Hidden 2 -> Classes
        self.out = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation pass:
        Transforms input tensor x through the sequence of linear combinations and activations.
        Returns raw logits of shape [batch_size, num_classes].
        """
        # Block 1
        x = self.fc1(x)
        if not self.training or x.size(0) > 1:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.fc2(x)
        if not self.training or x.size(0) > 1:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Output logits (CrossEntropyLoss takes raw logits directly)
        logits = self.out(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes softmax probabilities across the output classes.
        Sum of probabilities along last dimension equals 1.0.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=-1)
        return probabilities

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns architectural metadata and trainable parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }
