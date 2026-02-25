import os
import sys
import base64
import json
import re
import cv2
import gc
import logging
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from moviepy import VideoFileClip
from faster_whisper import WhisperModel

# ============================================================================
# ALIGNED CONFIGURATION
# ============================================================================

# Aligned with Visual Agent input/output structure
INPUT_VIDEO_DIR = os.environ.get("INPUT_VIDEO_DIR", "/workspace/qwen/agents/roadshow/test")

# FIX 1: Path aligned with ReasoningAgent's expectation
OUTPUT_DIR = os.environ.get("AUDIO_RESULTS_DIR", "/workspace/qwen/agents/roadshow/test_result/transcripts")
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AudioAgent")

# Create output dir immediately
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# UTILITIES
# ============================================================================

def get_device_config() -> Tuple[str, str]:
    """Determine hardware capabilities."""
    if torch.cuda.is_available():
        logger.info("CUDA detected. Switching to GPU acceleration.")
        return "cuda", "float16"
    logger.info("CUDA not detected. Using CPU.")
    return "cpu", "float32"


def get_video_files(folder_path: str, video_filter: Optional[List[str]] = None) -> List[Path]:
    """FIX 2: Enhanced to find files even if extensions are omitted in the command line."""
    p = Path(folder_path)
    if not p.exists():
        logger.error(f"Input directory does not exist: {folder_path}")
        return []

    if video_filter:
        files = []
        for v in video_filter:
            target = p / v
            # If 'test4' passed, check for 'test4.mp4', 'test4.mkv', etc.
            if not target.exists():
                for ext in VIDEO_EXTENSIONS:
                    candidate = p / f"{v}{ext}"
                    if candidate.exists():
                        target = candidate
                        break
            
            if target.exists() and target.is_file():
                files.append(target)
            else:
                logger.warning(f"File not found in {folder_path}: {v}")
        return files

    return sorted([
        f for f in p.iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS
    ])


# ============================================================================
# CORE PROCESSING
# ============================================================================

def extract_audio_safe(video_path: Path, audio_output_path: Path):
    """Robust extraction aligned with moviepy standards."""
    try:
        with VideoFileClip(str(video_path)) as clip:
            if clip.audio is None:
                raise ValueError(f"No audio track in {video_path.name}")

            clip.audio.write_audiofile(
                str(audio_output_path),
                fps=16000,
                nbytes=2,
                codec="pcm_s16le",
                logger=None,
                bitrate="192k"
            )
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        raise


def transcribe(audio_path: str, model: WhisperModel) -> Dict:
    """Transcribe with VAD filtering to prevent hallucinations in silence."""
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    return {
        "language": info.language,
        "segments": [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
        ]
    }


def save_results(result: Dict, video_path: Path, output_dir: Path):
    """FIX 3: Saves results using the exact naming pattern reasoning_agent expects."""
    base_name = video_path.stem

    # Save JSON for the Reasoning Agent (suffix must be _transcript.json)
    json_path = output_dir / f"{base_name}_transcript.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Save TXT for human review
    txt_path = output_dir / f"{base_name}_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Source: {video_path.name}\nLanguage: {result['language']}\n\n")
        for seg in result["segments"]:
            f.write(f"[{seg['start']:>7.2f}s -> {seg['end']:>7.2f}s] {seg['text']}\n")

    return json_path


# ============================================================================
# MAIN
# ============================================================================

def main(video_filter: Optional[List[str]] = None):
    device, compute_type = get_device_config()

    logger.info("-" * 50)
    logger.info(f"Target Input:  {INPUT_VIDEO_DIR}")
    logger.info(f"Target Output: {OUTPUT_DIR}")
    logger.info("-" * 50)

    video_files = get_video_files(INPUT_VIDEO_DIR, video_filter)
    if not video_files:
        logger.error("No valid video files found. Process aborted.")
        return 1

    logger.info(f"Loading Whisper {MODEL_SIZE}...")
    model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)

    success_count = 0

    for idx, video_path in enumerate(video_files, 1):
        logger.info(f"[{idx}/{len(video_files)}] Processing: {video_path.name}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_wav = Path(tmp_dir) / "extract.wav"

            try:
                extract_audio_safe(video_path, temp_wav)
                result = transcribe(str(temp_wav), model)
                json_file = save_results(result, video_path, Path(OUTPUT_DIR))

                logger.info(f"✓ Saved result to: {json_file.name}")
                success_count += 1

            except Exception:
                logger.error(f"✗ Failed: {video_path.name}")
                logger.debug(traceback.format_exc())

            finally:
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

    logger.info("-" * 50)
    logger.info(f"FINISHED. Processed {success_count} videos.")
    logger.info("-" * 50)
    return 0


if __name__ == "__main__":
    v_filter = sys.argv[1:] if len(sys.argv) > 1 else None
    sys.exit(main(v_filter))