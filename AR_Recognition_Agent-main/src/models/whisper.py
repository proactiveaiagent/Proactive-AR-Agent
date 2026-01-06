
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from .base import MultimodalModel
from typing import Optional
from dataclasses import dataclass

@dataclass
class WhisperConfig:
    model_name_or_path: str = "openai/whisper-larg-v3"  # placeholder; set to your qwen3-vl
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: str = "auto"  # "auto" or a torch dtype name

class Whisper(MultimodalModel):
    def __init__(self, cfg: WhisperConfig):
        dtype = None
        if cfg.torch_dtype != "auto":
            dtype = getattr(torch, cfg.torch_dtype)
            
        self.processor = AutoProcessor.from_pretrained(cfg.model_name_or_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            cfg.model_name_or_path,
            dtype=dtype,
            device_map="auto" if cfg.device.startswith("cuda") else None,
        )
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            dtype=cfg.torch_dtype,
            device_map="auto" if cfg.device.startswith("cuda") else None,
        )
        self.generate_kwargs = {
            "max_new_tokens": 256,
            "num_beams": 1,
            "condition_on_prev_tokens": False,
            "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "return_timestamps": True,
        }


    def generate(self, audio_path: str) -> Optional[str]:
        """Transcribe audio using the ASR model."""
        transcription = self.transcriber(audio_path, generate_kwargs=self.generate_kwargs)
        return transcription["text"] if transcription else None
    