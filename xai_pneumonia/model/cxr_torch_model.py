import sys
import types
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


class CXRFeatureExtractor(nn.Module):
    def __init__(self, weights: str = "densenet121-res224-all", cache_dir: str = None) -> None:
        super().__init__()
        if "lzma" not in sys.modules:
            lzma_stub = types.ModuleType("lzma")

            class _UnavailableLZMAFile:
                def __init__(self, *args, **kwargs) -> None:
                    raise RuntimeError("lzma is unavailable in this Python build.")

            lzma_stub.LZMAFile = _UnavailableLZMAFile
            lzma_stub.open = _UnavailableLZMAFile
            sys.modules["lzma"] = lzma_stub

        try:
            import torchxrayvision as xrv
        except ImportError as exc:
            raise ImportError(
                "torchxrayvision is required for CXR-pretrained encoders. "
                "Install it with: pip install torchxrayvision torchvision"
            ) from exc

        if cache_dir is None:
            cache_dir = str(Path(__file__).resolve().parents[2] / "outputs" / "torchxrayvision")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.encoder = xrv.models.DenseNet(weights=weights, cache_dir=cache_dir)
        self.feature_dim = int(getattr(self.encoder.classifier, "in_features", 1024))

    def freeze(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def unfreeze_last_block(self) -> None:
        self.freeze()
        for name, parameter in self.encoder.named_parameters():
            if name.startswith("features.denseblock4") or name.startswith("features.norm5"):
                parameter.requires_grad = True
        self.encoder.train()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder.features(images)
        features = torch.relu(features)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return torch.flatten(features, 1)


class CXRMLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class CXRClassifier(nn.Module):
    def __init__(self, feature_extractor: CXRFeatureExtractor, head: CXRMLPHead) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def freeze_encoder(self) -> None:
        self.feature_extractor.freeze()

    def unfreeze_last_block(self) -> None:
        self.feature_extractor.unfreeze_last_block()

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        return self.head(features)
