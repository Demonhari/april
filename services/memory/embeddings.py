from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError


class HashedTokenEmbedding(EmbeddingProvider):
    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in self._tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if math.isclose(norm, 0.0):
            return vector
        return vector / norm

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())


class LocalModelEmbeddingPlaceholder(EmbeddingProvider):
    @property
    def dimensions(self) -> int:
        raise NotImplementedError("Local GGUF embedding provider is an extension point.")

    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError("Local GGUF embedding provider is not implemented in the MVP.")
