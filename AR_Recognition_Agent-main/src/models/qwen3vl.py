from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLVideoProcessor
from qwen_vl_utils import process_vision_info
from .base import MultimodalInput, MultimodalModel


@dataclass
class Qwen3VLConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # placeholder; set to your qwen3-vl
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: str = "auto"  # "auto" or a torch dtype name
    max_new_tokens: int = 2048
    attn_implementation: str = "flash_attention_2"
    max_pixels: int = 50176


class Qwen3VLModel(MultimodalModel):
    """Qwen3-VL implementation via transformers.

    Notes:
    - Different Qwen-VL checkpoints may require slightly different processor/chat templates.
    - This wrapper keeps the rest of the codebase stable.
    """

    def __init__(self, cfg: Qwen3VLConfig):
        self.cfg = cfg

        dtype = None
        if cfg.torch_dtype != "auto":
            dtype = getattr(torch, cfg.torch_dtype)

        self.processor = AutoProcessor.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model_name_or_path,
            dtype=dtype,
            device_map="auto" if cfg.device.startswith("cuda") else None,
            trust_remote_code=True,
            attn_implementation=cfg.attn_implementation,
        )
        if not cfg.device.startswith("cuda"):
            self.model.to(cfg.device)

        self.model.eval()

    def generate(self, *, system_prompt: str, user_prompt: str, mm_input: MultimodalInput) -> str:
        images: Sequence[Image.Image] = mm_input.images or ()

        # Minimal, generic multi-image prompt. Adjust to match your checkpoint's expected format.
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img, "max_pixels": self.cfg.max_pixels} for img in images],
                    {"type": "text", "text": user_prompt},
                    *(
                        [{"type": "text", "text": f"ASR: {mm_input.audio_transcript}"}]
                        if mm_input.audio_transcript
                        else []
                    ),
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(text=prompt, images=images, videos=videos, do_resize=False, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens)

        # Decode only newly generated tokens.
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        return text.strip()
