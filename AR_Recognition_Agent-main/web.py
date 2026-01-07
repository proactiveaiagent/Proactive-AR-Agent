from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import gradio as gr

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.agent.ar_agent import ARRecognitionAgent
from src.memory.chroma_memory import ChromaUserMemory, MemoryConfig
from src.models.base import MultimodalInput
from src.models.qwen3vl import Qwen3VLConfig, Qwen3VLModel
from src.models.whisper import Whisper, WhisperConfig
from src.pipeline.preprocess import (
    FrameSamplingConfig,
    sample_video_frames,
    transcribe_audio_stub,
)

# ============================================================
# 🔒 固定配置（启动时决定，网页不可改）
# ============================================================
USER_ID = "demo_user"

# Models
WHISPER_MODEL_NAME = "openai/whisper-large-v3"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Qwen config
ATTN_IMPLEMENTATION = "sdpa"
TORCH_DTYPE = "bfloat16"     # "auto" | "float16" | "bfloat16"
MAX_PIXELS = 262144         # 512x512 pixels

# Memory / Embedding
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "chroma_db"


# ============================================================
# Global Singleton Models
# ============================================================
WHISPER_MODEL: Optional[Whisper] = None
QWEN_MODEL: Optional[Qwen3VLModel] = None
MEMORY: Optional[ChromaUserMemory] = None


# ============================================================
# Initialization (runs once at page load)
# ============================================================
def init_all():
    global WHISPER_MODEL, QWEN_MODEL, MEMORY

    print("🚀 Initializing models and memory...")

    if WHISPER_MODEL is None:
        print("  - Loading Whisper")
        WHISPER_MODEL = Whisper(
            WhisperConfig(
                model_name_or_path=WHISPER_MODEL_NAME
            )
        )

    if QWEN_MODEL is None:
        print("  - Loading Qwen3-VL")
        QWEN_MODEL = Qwen3VLModel(
            Qwen3VLConfig(
                model_name_or_path=QWEN_MODEL_NAME,
                attn_implementation=ATTN_IMPLEMENTATION,
                torch_dtype=TORCH_DTYPE,
                max_pixels=MAX_PIXELS,
            )
        )

    if MEMORY is None:
        print("  - Initializing Chroma Memory")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL
        )
        vs = Chroma(
            collection_name=MemoryConfig.collection_name,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
        MEMORY = ChromaUserMemory(
            vectorstore=vs,
            cfg=MemoryConfig(persist_directory=PERSIST_DIR),
        )

    print("✅ Initialization complete")


# ============================================================
# Core Inference Pipeline
# ============================================================
def run_pipeline(
    video_path: str,
    audio_path: Optional[str],
):
    global WHISPER_MODEL, QWEN_MODEL, MEMORY

    if video_path is None:
        raise gr.Error("Please upload a video")

    # 1. Sample video frames
    frames = sample_video_frames(
        video_path,
        FrameSamplingConfig(),
    )

    # 2. ASR (optional)
    transcript = None
    if audio_path is not None:
        transcript = transcribe_audio_stub(
            audio_path,
            WHISPER_MODEL,
        )

    # 3. Multimodal input
    mm_input = MultimodalInput(
        images=frames,
        audio_transcript=transcript,
        metadata={"video": video_path},
    )

    # 4. Agent
    agent = ARRecognitionAgent(
        model=QWEN_MODEL,
        memory=MEMORY,
    )

    scene = agent.run(
        user_id=USER_ID,
        mm_input=mm_input,
    )

    scene_dict = scene.model_dump()
    scene_json = json.dumps(
        scene_dict,
        ensure_ascii=False,
        indent=2,
    )

    return scene_dict, scene_json


# ============================================================
# Gradio UI (Upload-only)
# ============================================================
with gr.Blocks(title="AR Scene Recognition Agent") as demo:
    gr.Markdown("## 🧠 AR Multimodel")

    # 🔥 Load models at page startup
    demo.load(
        fn=init_all,
        inputs=[],
        outputs=[],
    )

    with gr.Row():
        video = gr.Video(
            label="上传视频",
            sources=["upload"],
        )
        audio = gr.Audio(
            label="上传音频（可选）",
            sources=["upload"],
            type="filepath",
        )

    run_btn = gr.Button(
        "🚀 开始场景理解",
        variant="primary",
    )

    gr.Markdown("### 📤 输出结果")

    with gr.Tabs():
        with gr.Tab("结构化结果"):
            scene_view = gr.JSON()

        with gr.Tab("Raw JSON"):
            scene_text = gr.Code(
                language="json",
            )

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            video,
            audio,
        ],
        outputs=[
            scene_view,
            scene_text,
        ],
    )


if __name__ == "__main__":
    demo.launch()
