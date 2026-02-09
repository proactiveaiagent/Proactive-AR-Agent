# visual_agent_with_rag.py
import os
from pathlib import Path
import base64
import tempfile
import shutil
import cv2
import gc
import json
from typing import List, Dict, Optional
import random
import asyncio
import numpy as np

from openai import OpenAI

# RAG-Anything imports
from raganything import RAGAnything, RAGAnythingConfig

# LightRAG imports for LLM/embedding functions
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Sentence transformers for local embeddings
from sentence_transformers import SentenceTransformer

# Import memory system
from memory_store import find_by_image_embedding, get_memory_by_id
from clip_embed import get_image_embedding

# ---- OpenAI / local Qwen-VL client config ----
openai_api_base = "http://127.0.0.1:25000/v1"
client = OpenAI(base_url=openai_api_base, api_key="EMPTY")

# ---- Video input / preprocessing config ----
INPUT_VIDEO_DIR = "/workspace/test/test_data"
log_dir = "/workspace/test/visual_results"
OUTPUT_DIR = "/workspace/test/visual_results"
FRAME_SAMPLE_FPS = 1
BATCH_SIZE = 10

os.makedirs(log_dir, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- RAG-Anything initialization ----
RAG_WORKING_DIR = "./rag_storage"

# API configuration for RAG's LLM/embedding functions
api_key = "EMPTY"
base_url = openai_api_base

# Define LLM model function for RAG
def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return openai_complete_if_cache(
        "/models/Qwen2.5-VL-7B-Instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )

# Define vision model function for RAG
def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
    """
    Vision model function that RAG-Anything will call for image analysis.
    Supports both messages format (multimodal) and single image_data format.
    """
    if messages:
        # Multimodal messages format (VLM enhanced query)
        return openai_complete_if_cache(
            "/models/Qwen2.5-VL-7B-Instruct",
            "",
            system_prompt=None,
            history_messages=[],
            messages=messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    elif image_data:
        # Single image format
        return openai_complete_if_cache(
            "/models/Qwen2.5-VL-7B-Instruct",
            "",
            system_prompt=None,
            history_messages=[],
            messages=[
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        },
                    ],
                } if image_data else {"role": "user", "content": prompt},
            ],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    else:
        # Pure text fallback
        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

# Global embedding model
_embedding_model = None

# Local model path config
HF_LOCAL_MODELS_DIR = "/workspace/qwen/agents/RAG_agents/models"
# local_subdir should match the folder name you created with wget/unpack
LOCAL_EMBED_SUBDIR = "paraphrase-multilingual-MiniLM-L12-v2"

def get_embedding_model():
    """
    Load embedding model from a local directory (no network).
    Ensure you have downloaded the model into:
      HF_LOCAL_MODELS_DIR / LOCAL_EMBED_SUBDIR
    Example: /workspace/qwen/agents/RAG_agents/models/paraphrase-multilingual-MiniLM-L12-v2
    """
    global _embedding_model
    if _embedding_model is None:
        local_path = os.path.join(HF_LOCAL_MODELS_DIR, LOCAL_EMBED_SUBDIR)
        if not os.path.isdir(local_path) or not os.listdir(local_path):
            raise RuntimeError(
                f"Local embedding model not found at {local_path}. "
                "Please download and unpack the model there before running."
            )
        # Force offline load
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print("Loading embedding model from local path:", local_path)
        _embedding_model = SentenceTransformer(local_path, device="cpu")
        print("✓ Embedding model loaded from", local_path)
    return _embedding_model

async def local_embed_func(texts):
    """
    Local embedding function using SentenceTransformer.
    Fixed to handle input/output shape properly.
    """
    if not texts:
        return []
    
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]
    
    model = get_embedding_model()
    
    # Encode with explicit parameters
    embeddings = model.encode(
        texts,
        batch_size=len(texts),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    
    # Ensure correct shape: (len(texts), embedding_dim)
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    
    # Debug: Verify output matches input count
    if embeddings.shape[0] != len(texts):
        print(f"WARNING: Embedding count mismatch!")
        print(f"  Input texts count: {len(texts)}")
        print(f"  Output embeddings shape: {embeddings.shape}")
        print(f"  Input texts: {texts[:3]}...")  # Show first 3
        
        # Attempt to fix by taking mean if multiple embeddings returned
        if embeddings.shape[0] > len(texts):
            print(f"  Attempting to fix by averaging extra embeddings...")
            fixed_embeddings = []
            for i in range(len(texts)):
                start_idx = i * (embeddings.shape[0] // len(texts))
                end_idx = start_idx + (embeddings.shape[0] // len(texts))
                fixed_embeddings.append(embeddings[start_idx:end_idx].mean(axis=0))
            embeddings = np.array(fixed_embeddings)
            print(f"  Fixed shape: {embeddings.shape}")
    
    return embeddings.tolist()

# Define embedding function for RAG
embedding_func = EmbeddingFunc(
    embedding_dim=384,  # paraphrase-multilingual-MiniLM-L12-v2 dimension
    max_token_size=512,
    func=local_embed_func,
)

# Create RAG instance (synchronous wrapper around async RAGAnything init)
rag = None

def init_rag_sync():
    global rag
    config = RAGAnythingConfig(
        working_dir=RAG_WORKING_DIR,
        parser=None,
        enable_image_processing=True,
        enable_table_processing=False,
        enable_equation_processing=False,
    )
    
    async def _init():
        global rag
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func
        )
    
    asyncio.run(_init())

# ---- Video utilities ----
def get_video_files(directory):
    p = Path(directory)
    if not p.exists():
        raise RuntimeError(f"Directory does not exist: {directory}")
    videos = sorted(list(p.glob("*.mp4")))
    if not videos:
        raise RuntimeError(f"No .mp4 files found in {directory}")
    return videos

def extract_frames_from_video(video_path, output_folder, sample_fps=1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for frame extraction: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(video_fps / max(1e-6, sample_fps))))
    os.makedirs(output_folder, exist_ok=True)
    frame_paths = []
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_paths.append(frame_filename)
            saved_count += 1
        frame_count += 1
    cap.release()
    return frame_paths

def image_to_data_url(file_path):
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(file_path).suffix.lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{encoded}"

def torch_br_gc():
    gc.collect()

def check_memory_matches(frame_paths: List[str], video_name: str) -> List[Dict]:
    """Check if any frames match known entities in memory"""
    try:
        sample_size = min(3, len(frame_paths))
        sampled_frames = random.sample(frame_paths, sample_size)
        matches = []
        for frame_path in sampled_frames:
            vec = get_image_embedding(frame_path)
            results = find_by_image_embedding(vec, top_k=3, threshold=0.75)
            for mem in results:
                matches.append({
                    "memory_id": mem["id"],
                    "label": mem["canonical_label"],
                    "type": mem["type"],
                    "similarity": mem["similarity_score"],
                    "frame": frame_path
                })
        return matches
    except Exception as e:
        print(f"  Warning: Memory check failed: {e}")
        return []

# ---- Async helper to insert content_list into RAG ----
async def async_insert_content_list(rag_instance: RAGAnything, content_list, file_path, doc_id=None):
    """Uses RAG-Anything API to insert parsed content list"""
    await rag_instance.insert_content_list(
        content_list=content_list,
        file_path=file_path,
        split_by_character=None,
        split_by_character_only=False,
        doc_id=doc_id,
        display_stats=False
    )

# ---- Main processing loop ----
def process_videos_and_run_batched():
    """Main entry point for video processing with RAG integration"""
    print("="*70)
    print("VISUAL AGENT WITH RAG")
    print("="*70)
    
    # Ensure RAG initialized
    print("Initializing RAG...")
    init_rag_sync()
    print("✓ RAG initialized")
    
    global rag

    video_files = get_video_files(INPUT_VIDEO_DIR)
    print(f"\nFound {len(video_files)} video(s) to process\n")

    for vid_idx, video_path in enumerate(video_files):
        video_name = video_path.stem
        print(f"\n{'='*70}")
        print(f"Processing video {vid_idx+1}/{len(video_files)}: {video_name}")
        print(f"{'='*70}")
        
        tmp_frames_dir = None
        try:
            # Extract frames
            print("  [1/4] Extracting frames...")
            tmp_frames_dir = tempfile.mkdtemp(prefix=f"frames_{video_name}_")
            frame_paths = extract_frames_from_video(str(video_path), tmp_frames_dir, sample_fps=FRAME_SAMPLE_FPS)
            
            if not frame_paths:
                print(f"  ✗ No frames extracted for {video_name}, skipping.")
                continue
            
            print(f"  ✓ Extracted {len(frame_paths)} frames")

            # Check memory for known entities
            print("\n  [2/4] Checking memory for known entities...")
            memory_matches = check_memory_matches(frame_paths, video_name)
            if memory_matches:
                print(f"  ✓ Found {len(memory_matches)} memory match(es):")
                for match in memory_matches[:3]:
                    print(f"    - {match['label']} ({match['type']}) similarity: {match['similarity']:.3f}")
            else:
                print(f"  No known entities recognized")

            # Process frames in batches
            print(f"\n  [3/4] Processing {len(frame_paths)} frames in batches of {BATCH_SIZE}...")
            total_frames = len(frame_paths)
            batch_count = 0
            
            for b_start in range(0, total_frames, BATCH_SIZE):
                b_end = min(b_start + BATCH_SIZE, total_frames)
                batch_paths = frame_paths[b_start:b_end]
                batch_count += 1
                
                print(f"\n    Batch {batch_count}: frames {b_start}..{b_end-1}")

                try:
                    # Build content list for Qwen-VL call
                    request_content = [{
                        "type": "text",
                        "text": (
                            "You are analyzing first-person AR glasses footage. Provide detailed objective visual observations in ONE JSON object (no arrays, no extra text). "
                            "Include: scene_description, objects_visible, people_present, activities_observed, spatial_layout, temporal_progression."
                        )
                    }]
                    
                    for p in batch_paths:
                        data_url = image_to_data_url(p)
                        request_content.append({"type": "image_url", "image_url": {"url": data_url}})

                    # Call local Qwen-VL
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": request_content}],
                        model="/models/Qwen2.5-VL-7B-Instruct",
                        temperature=0,
                        max_tokens=4096
                    )
                    result_text = chat_completion.choices[0].message.content

                    # Parse model output as JSON
                    parsed_obj = None
                    try:
                        parsed = json.loads(result_text)
                        parsed_obj = parsed if isinstance(parsed, dict) else {"model_output": parsed}
                    except Exception:
                        parsed_obj = {"raw_text": result_text}

                    # Add metadata
                    parsed_obj["video"] = video_name
                    parsed_obj["frame_start"] = b_start
                    parsed_obj["frame_end"] = b_end - 1
                    parsed_obj["batch_frame_count"] = len(batch_paths)
                    
                    if memory_matches:
                        parsed_obj["memory_context"] = memory_matches

                    # Save JSON file per batch
                    out_filename = os.path.join(log_dir, f"{video_name}_frames_{b_start:06d}_{b_end-1:06d}.json")
                    with open(out_filename, "w", encoding="utf-8") as f:
                        json.dump(parsed_obj, f, ensure_ascii=False, indent=2)
                    print(f"    ✓ Saved: {out_filename}")

                    # Build RAG content_list
                    print(f"    [4/4] Inserting batch into RAG...")
                    rag_content_list = []
                    
                    # Insert the parsed JSON as a text entry
                    rag_content_list.append({
                        "type": "text",
                        "text": json.dumps(parsed_obj, ensure_ascii=False),
                        "page_idx": b_start
                    })
                    
                    # Insert each frame as an image entry
                    for i, p in enumerate(batch_paths):
                        rag_content_list.append({
                            "type": "image",
                            "img_path": os.path.abspath(p),
                            "image_caption": [f"{video_name} frame {b_start + i}"],
                            "page_idx": b_start + i,
                        })

                    # Call RAG insertion
                    try:
                        asyncio.run(async_insert_content_list(rag, rag_content_list, file_path=str(video_path)))
                        print(f"    ✓ Inserted batch into RAG")
                    except Exception as ie:
                        print(f"    ✗ Error inserting into RAG: {ie}")

                except Exception as e:
                    print(f"    ✗ Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()

                torch_br_gc()

            print(f"\n✓ Completed processing {video_name}")

        except Exception as e:
            print(f"\n✗ Error processing video {video_name}: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Clean up temporary frames directory
            try:
                if tmp_frames_dir and os.path.exists(tmp_frames_dir):
                    shutil.rmtree(tmp_frames_dir)
            except Exception as e:
                print(f"  Warning: Failed to cleanup temp directory: {e}")

    print("\n" + "="*70)
    print("VISUAL PROCESSING COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    process_videos_and_run_batched()