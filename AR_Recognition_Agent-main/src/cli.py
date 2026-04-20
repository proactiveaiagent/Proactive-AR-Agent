from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .agent.ar_agent import ARRecognitionAgent
from .memory.chroma_memory import ChromaUserMemory, MemoryConfig
from .models.base import MultimodalInput
from .models.qwen3vl import Qwen3VLConfig, Qwen3VLModel
from .models.whisper import Whisper, WhisperConfig
from .pipeline.preprocess import FrameSamplingConfig, sample_video_frames, transcribe_audio_stub, face_detect


def build_memory(persist_dir: str, embed_model: str) -> ChromaUserMemory:
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vs = Chroma(
        collection_name=MemoryConfig.collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir + "/object_memory",
    )
    pvs = Chroma(
        collection_name=MemoryConfig.people_collection_name,
        embedding_function=None,
        persist_directory=persist_dir + "/people_memory",
    )
    
    return ChromaUserMemory(vectorstore=vs, cfg=MemoryConfig(persist_directory=persist_dir), people_vectorstore=pvs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--persist-dir", default="chroma_db")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--asr-model", default="openai/whisper-large-v3")
    ap.add_argument("--out", default="outputs/scene.json")
    ap.add_argument("--attn_implementation", default="sdpa")
    ap.add_argument("--torch_dtype", default="auto", help="Torch dtype for model loading, e.g., 'float16', 'bfloat16', or 'auto'")
    ap.add_argument("--max_pixels", type=int, default=50176, help="Maximum number of pixels for image processing")
    args = ap.parse_args()

    Path("outputs").mkdir(parents=True, exist_ok=True)

    frames = sample_video_frames(args.video, FrameSamplingConfig())
    frames, faces, ann_img = face_detect(frames)
    if args.audio:
        asr_model = Whisper(WhisperConfig(model_name_or_path=args.asr_model))
        transcript = transcribe_audio_stub(args.audio, asr_model)
    else:
        transcript = None

    mm_input = MultimodalInput(images=frames, faces=faces, audio_transcript=transcript, metadata={"video": args.video}, ann_img=ann_img)

    memory = build_memory(args.persist_dir, args.embed_model)
    model = Qwen3VLModel(
        Qwen3VLConfig(
            model_name_or_path=args.model, 
            attn_implementation=args.attn_implementation, 
            torch_dtype=args.torch_dtype,
            max_pixels=args.max_pixels
        )
    )

    agent = ARRecognitionAgent(model=model, memory=memory)
    scene = agent.run(user_id=args.user_id, mm_input=mm_input)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(scene.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
