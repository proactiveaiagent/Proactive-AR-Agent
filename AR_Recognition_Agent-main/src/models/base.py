from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from PIL import Image


@dataclass
class MultimodalInput:
    """Normalized multimodal input.

    - `images`: sequence of PIL images (e.g., sampled video frames)
    - `faces`: dict mapping people_id to face embeddings for face-based retrieval
    - `audio_transcript`: ASR text if available
    - `metadata`: any additional extracted signals (timestamps, detected sound events, etc.)
    """

    images: Sequence[Image.Image] = ()
    ann_img: Optional[Image.Image] = None
    faces: Optional[dict[str, Any]] = None  # 可以存储 {people_id: face_embedding_vector}
    audio_transcript: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class MultimodalModel(ABC):
    """Abstract base class for multimodal LLMs.

    Implementations should:
    - accept `MultimodalInput`
    - return a *string* which is the model raw output (ideally JSON)
    """

    @abstractmethod
    def generate(self, *, system_prompt: str, user_prompt: str, mm_input: MultimodalInput) -> str:
        raise NotImplementedError


class SupportsWarmup(ABC):
    @abstractmethod
    def warmup(self) -> None:
        """Optional: load model weights to GPU/CPU and run a tiny pass."""
        raise NotImplementedError
