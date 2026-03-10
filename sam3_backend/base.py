"""Abstract base class for SAM backends."""

from abc import ABC, abstractmethod
from pathlib import Path


class SAMBackend(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend framework is importable and weights exist."""
        ...

    @abstractmethod
    def load(self, model_dir: Path) -> None:
        """Load model weights from model_dir into memory."""
        ...

    @abstractmethod
    def encode_image(self, image_path: Path) -> str:
        """
        Run the image encoder and cache the embedding.

        Returns the embedding_id (equals the file UUID / filename stem).
        """
        ...

    @abstractmethod
    def predict(
        self,
        embedding_id: str,
        prompts: list[dict],
        width: int,
        height: int,
    ) -> list[dict]:
        """
        Run the mask decoder for the given prompts on a cached embedding.

        prompts items:
            {"type": "point", "x": 0-1, "y": 0-1, "polarity": 1|-1}
            {"type": "box", "x1": 0-1, "y1": 0-1, "x2": 0-1, "y2": 0-1}

        Returns a list of mask dicts:
            {"segmentation": [x1,y1,x2,y2,...], "bbox": [x,y,w,h], "score": float}

        All coordinates are normalized 0-1.
        """
        ...

    @abstractmethod
    def predict_text(
        self,
        embedding_id: str,
        text_prompt: str,
        width: int,
        height: int,
        confidence_threshold: float = 0.3,
    ) -> list[dict]:
        """
        Text-grounded mask prediction on a cached embedding.

        Returns: [{"segmentation": [x1,y1,...], "bbox": [x,y,w,h], "score": float, "label": str}]
        All coordinates normalized 0-1.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model weights from memory."""
        ...


class UnavailableBackend(SAMBackend):
    """Fallback backend used when the required ML framework is not installed."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def is_available(self) -> bool:
        return False

    def load(self, model_dir: Path) -> None:
        raise RuntimeError(f"SAM backend unavailable: {self._reason}")

    def encode_image(self, image_path: Path) -> str:
        raise RuntimeError(f"SAM backend unavailable: {self._reason}")

    def predict(self, embedding_id, prompts, width, height) -> list[dict]:
        raise RuntimeError(f"SAM backend unavailable: {self._reason}")

    def predict_text(self, embedding_id, text_prompt, width, height, confidence_threshold=0.3) -> list[dict]:
        raise RuntimeError(f"SAM backend unavailable: {self._reason}")

    def unload(self) -> None:
        pass
