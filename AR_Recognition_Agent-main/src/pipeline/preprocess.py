from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
from PIL import Image
from ..models.base import MultimodalModel


@dataclass
class FrameSamplingConfig:
    fps: float = 1.0
    max_frames: int = 16


def sample_video_frames(video_path: str, cfg: FrameSamplingConfig) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(int(round(video_fps / max(cfg.fps, 0.001))), 1)

    frames: List[Image.Image] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            if len(frames) >= cfg.max_frames:
                break
        idx += 1

    cap.release()
    return frames


def transcribe_audio_stub(audio_path: str, asr_model: MultimodalModel) -> Optional[str]:
    """Placeholder for ASR.

    Implement with Whisper or your ASR service, returning a text transcript.
    """

    transcription = asr_model.generate(audio_path)
    return transcription
