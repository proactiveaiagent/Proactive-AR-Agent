"""
api_server.py — Qwen3-VL + Whisper API Server (fixed)
=======================================================
Fixes vs original:
  1. torch_dtype (not dtype) in from_pretrained
  2. Per-request temp files for /analyze (no race condition)
  3. asyncio.Lock around generate() calls to prevent concurrent OOM
  4. torch.cuda.empty_cache() after each generate
  5. lifespan replaces deprecated @app.on_event("startup")
  6. Temp image cleanup in /analyze via try/finally
  7. /health reports GPU memory usage for debugging
"""

import os
import io
import shutil
import tempfile
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import uvicorn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

qwen_model = None
processor = None
whisper_model = None

# Prevents concurrent generate() calls from OOM-ing the GPU
_generate_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qwen_model, processor, whisper_model

    # ── Qwen3-VL ────────────────────────────────────────────────────────────
    print("🚀 Loading Qwen3-VL model...")
    model_path = "/hpc2hdd/home/jyinap/Proactive_Agent/models/Qwen3-VL-4B-Instruct"

    qwen_model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,   # ← fixed: was `dtype`
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print("✅ Qwen3-VL ready!")

    # ── faster-whisper ───────────────────────────────────────────────────────
    print("🚀 Loading Whisper model...")
    from faster_whisper import WhisperModel

    whisper_size = os.environ.get("WHISPER_MODEL_SIZE", "medium")
    w_device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if w_device == "cuda" else "float32"

    whisper_model = WhisperModel(whisper_size, device=w_device, compute_type=compute_type)
    print(f"✅ Whisper ({whisper_size}) ready on {w_device} ({compute_type})")

    yield  # server runs here

    # ── Shutdown cleanup ─────────────────────────────────────────────────────
    print("🛑 Shutting down — freeing GPU memory...")
    del qwen_model, processor, whisper_model
    torch.cuda.empty_cache()


app = FastAPI(title="Qwen3-VL + Whisper API Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# /analyze  — vision + text
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze(image: UploadFile = File(...), prompt: str = Form(...)):
    """Analyze an image with a text prompt using Qwen3-VL."""
    # Write to a per-request temp file to avoid race conditions
    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        with os.fdopen(tmp_fd, "wb") as f:
            pil_image.save(f, format="JPEG")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": tmp_path},
                {"type": "text",  "text": prompt}
            ]
        }]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(qwen_model.device)

        async with _generate_lock:          # ← serialise GPU calls
            generated_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=2048,
                top_p=0.8,
                top_k=20,
                temperature=0.3,
                repetition_penalty=1.0
            )
            torch.cuda.empty_cache()        # ← free fragmented memory

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return JSONResponse(content={"analysis": output[0]})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):        # ← always clean up
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# /consolidate  — text-only
# ---------------------------------------------------------------------------

@app.post("/consolidate")
async def consolidate_memory(request: Request):
    """Memory consolidation — text-only Qwen call."""
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            return JSONResponse(content={"error": "No prompt provided"}, status_code=400)

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(qwen_model.device)

        async with _generate_lock:
            generated_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=4096,
                top_p=0.8,
                top_k=20,
                temperature=0.2,
                repetition_penalty=1.05
            )
            torch.cuda.empty_cache()

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return JSONResponse(content={"analysis": output[0]})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# /transcribe  — Whisper
# ---------------------------------------------------------------------------

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe audio with faster-whisper."""
    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        segments, info = whisper_model.transcribe(
            tmp_path,
            beam_size=5,
            vad_filter=False,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        transcript_segments = []
        full_text_parts = []
        for seg in segments:
            transcript_segments.append({
                "start": round(seg.start, 2),
                "end":   round(seg.end, 2),
                "text":  seg.text.strip()
            })
            full_text_parts.append(seg.text.strip())

        full_text = " ".join(full_text_parts)

        if not transcript_segments:
            return JSONResponse(content={
                "language": getattr(info, "language", "unknown"),
                "full_text": "",
                "segments": [],
                "no_speech": True
            })

        return JSONResponse(content={
            "language": info.language,
            "full_text": full_text,
            "segments": transcript_segments,
            "no_speech": False
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    info = {
        "status": "healthy",
        "model_loaded":   qwen_model is not None,
        "whisper_loaded": whisper_model is not None,
        "generate_locked": _generate_lock.locked(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        allocated = torch.cuda.memory_allocated(0)
        total = props.total_memory
        info["gpu"] = {
            "name": props.name,
            "total_gb":     round(total     / 1024**3, 2),
            "allocated_gb": round(allocated / 1024**3, 2),
            "free_gb":      round((total - allocated) / 1024**3, 2),
        }
    return info


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)