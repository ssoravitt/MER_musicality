"""Tabular SOTA controls (Panda et al. 2020 SVM & Louro et al. 2024 Fig. 4 DNN)."""

import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn


def build_panda_svm(C: float = 1.0, kernel: str = "rbf") -> Pipeline:
    """Standardized RBF-SVM reproducing Panda et al. (2020) tabular baseline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=C, kernel=kernel, probability=True, random_state=42)),
        ]
    )


class LouroTabularDNN(nn.Module):
    """Reproduction of Louro et al. (Sensors 2024, Fig. 4) tabular DNN for 100/1714 features."""

    def __init__(self, in_features: int, num_classes: int = 4, use_batchnorm: bool = True) -> None:
        super().__init__()
        hidden = (100, 50) if in_features <= 200 else (500, 300, 100)
        layers: list[nn.Module] = []
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(in_features))
        prev = in_features
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
