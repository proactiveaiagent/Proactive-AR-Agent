# transcribe_with_rag.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import asyncio
import numpy as np
from moviepy import VideoFileClip
from faster_whisper import WhisperModel
from tqdm import tqdm
from pathlib import Path

# RAG-Anything
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Sentence transformers for local embeddings
from sentence_transformers import SentenceTransformer

# CONFIG
input_folder = "/workspace/test/test_data"
output_folder = "/workspace/test/test_data/transcripts"
model_size = "medium"
RAG_WORKING_DIR = "./rag_storage"

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

# API configuration for RAG
api_key = "EMPTY"
base_url = "http://127.0.0.1:25000/v1"

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

# Global embedding model
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        print("✓ Embedding model loaded")
    return _embedding_model

async def local_embed_func(texts):
    """Local embedding function using SentenceTransformer"""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()

# Define embedding function for RAG
embedding_func = EmbeddingFunc(
    embedding_dim=384,  # paraphrase-multilingual-MiniLM-L12-v2 dimension
    max_token_size=512,
    func=local_embed_func,
)

# Initialize RAG (synchronous helper that runs minimal async setup)
rag = None

def init_rag():
    global rag
    config = RAGAnythingConfig(
        working_dir=RAG_WORKING_DIR,
        parser=None,
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )
    
    async def _init():
        global rag
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func
        )
    
    asyncio.run(_init())

async def async_insert_content_list(rag_instance: RAGAnything, content_list, file_path, doc_id=None):
    """Insert content list into RAG asynchronously"""
    await rag_instance.insert_content_list(
        content_list=content_list,
        file_path=file_path,
        split_by_character=None,
        split_by_character_only=False,
        doc_id=doc_id,
        display_stats=False
    )

def extract_audio(video_path: str, audio_path: str, fps: int = 16000):
    """Extract audio from video file"""
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise ValueError("No audio track found in video")
    clip.audio.write_audiofile(
        audio_path,
        fps=fps,
        nbytes=2,
        codec="pcm_s16le",
        logger=None,
        bitrate="192k"
    )
    clip.close()

def transcribe_faster_whisper(audio_path: str, model_size: str, model=None):
    """Transcribe audio using Faster Whisper"""
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    audio_clip = AudioFileClip(audio_path)
    total_duration = audio_clip.duration
    audio_clip.close()

    if model is None:
        model = WhisperModel(model_size, device="cpu", compute_type="float32")

    segments, info = model.transcribe(
        audio_path,
        language=None,
        beam_size=10,
        best_of=5,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=True
    )
    
    segments_list_raw = list(segments)
    segments_list = []
    
    for seg in segments_list_raw:
        segments_list.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        })

    result = {
        "language": info.language,
        "segments": segments_list
    }
    
    return result, model

def save_outputs(result, video_name: str, output_folder: str):
    """Save transcript outputs as TXT and JSON"""
    os.makedirs(output_folder, exist_ok=True)
    base_name = Path(video_name).stem
    txt_path = os.path.join(output_folder, f"{base_name}_transcript.txt")
    json_path = os.path.join(output_folder, f"{base_name}_transcript.json")

    # Save as TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            f.write(f"SEGMENT {seg['start']:.2f} --> {seg['end']:.2f} : {seg['text']}\n")

    # Save as JSON
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(result, jf, ensure_ascii=False, indent=2)
    
    return txt_path, json_path

def get_video_files(folder_path: str):
    """Get all video files from folder"""
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(Path(folder_path).glob(f"*{ext}"))
        video_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    return sorted(video_files)

def main():
    print("="*70)
    print("AUDIO TRANSCRIPTION WITH RAG")
    print("="*70)
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Model size:    {model_size}")
    print("="*70)
    
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    video_files = get_video_files(input_folder)
    if not video_files:
        print(f"No video files found in {input_folder}")
        return

    print(f"\nFound {len(video_files)} video(s) to process")
    print("\nLoading Whisper model...")
    model = WhisperModel(model_size, device="cpu", compute_type="float32")
    print(f"✓ Loaded Whisper model: {model_size}")

    # Init RAG
    print("\nInitializing RAG...")
    try:
        init_rag()
        print("✓ RAG initialized")
    except Exception as e:
        print(f"Warning: Failed to init RAG: {e}. Continuing without RAG.")
    
    global rag

    for idx, video_path in enumerate(video_files, 1):
        video_name = video_path.name
        print(f"\n{'='*70}")
        print(f"Processing [{idx}/{len(video_files)}]: {video_name}")
        print(f"{'='*70}")
        
        audio_path = None
        try:
            audio_path = os.path.join(input_folder, f"temp_audio_{idx}.wav")
            
            print("  [1/4] Extracting audio...")
            extract_audio(str(video_path), audio_path)
            print("  ✓ Audio extracted")

            print("  [2/4] Transcribing with Whisper...")
            result, model = transcribe_faster_whisper(audio_path, model_size, model)
            print(f"  ✓ Transcribed {len(result['segments'])} segments (language: {result.get('language', 'unknown')})")

            print("  [3/4] Saving outputs...")
            txt_path, json_path = save_outputs(result, video_name, output_folder)
            print(f"  ✓ Saved TXT: {txt_path}")
            print(f"  ✓ Saved JSON: {json_path}")
            
            # Clean up temp audio
            if os.path.exists(audio_path):
                os.remove(audio_path)
                audio_path = None

            # Build RAG content_list from segments
            if rag is not None:
                print("  [4/4] Inserting into RAG...")
                content_list = []
                
                # Add a summary entry for the whole transcript
                summary_text = {
                    "type": "text",
                    "text": json.dumps({
                        "video": video_name,
                        "language": result.get("language"),
                        "segment_count": len(result["segments"]),
                        "total_text": " ".join([seg["text"] for seg in result["segments"]])
                    }, ensure_ascii=False),
                    "page_idx": 0
                }
                content_list.append(summary_text)

                # Add each segment as a text content item
                for si, seg in enumerate(result["segments"]):
                    seg_entry = {
                        "type": "text",
                        "text": f"[{seg['start']:.2f}s-{seg['end']:.2f}s] {seg['text']}",
                        "page_idx": si + 1
                    }
                    content_list.append(seg_entry)

                # Insert into RAG
                try:
                    asyncio.run(async_insert_content_list(rag, content_list, file_path=str(video_path)))
                    print(f"  ✓ Inserted {len(content_list)} items into RAG")
                except Exception as e:
                    print(f"  ✗ Error inserting transcript into RAG: {e}")
            else:
                print("  [4/4] Skipping RAG insertion (RAG not initialized)")

            print(f"✓ SUCCESS: {video_name}")

        except Exception as e:
            print(f"✗ FAILED {video_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up temp audio on error
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass

    print("\n" + "="*70)
    print("AUDIO TRANSCRIPTION COMPLETE")
    print("="*70)
    print(f"All transcripts saved to: {output_folder}")
    print("="*70)

if __name__ == "__main__":
    main()