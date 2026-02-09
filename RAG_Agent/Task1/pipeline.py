#!/usr/bin/env python3
"""
Orchestrator: visual_agent_with_rag -> transcribe_with_rag -> Reason_agent_with_rag
"""

import os
import shutil
import tempfile
from pathlib import Path

# Test video (already in input directory)
TEST_VIDEO_PATH = "/workspace/test/test_data/test1.mp4"

# Import updated agents (update names if you used different filenames)
import visual_agent_with_rag as visual_agent
import transcribe_with_rag as audio_agent
import Reason_agent_with_rag as reason_agent

# Directories (keep in sync with agents)
VISUAL_INPUT_DIR = "/workspace/test/test_data"
VISUAL_RESULTS_DIR = "/workspace/test/visual_results"
AUDIO_INPUT_DIR = "/workspace/test/test_data"
AUDIO_OUTPUT_DIR = "/workspace/test/test_data/transcripts"
REASON_OUTPUT_DIR = "/workspace/test/reasoning_results"

# Ensure directories exist
os.makedirs(VISUAL_INPUT_DIR, exist_ok=True)
os.makedirs(VISUAL_RESULTS_DIR, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(REASON_OUTPUT_DIR, exist_ok=True)


def clear_directory(directory: str):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")


def run_visual_agent_for_video(video_path: str):
    temp_dir = tempfile.mkdtemp(prefix="visual_input_")
    temp_video_path = os.path.join(temp_dir, os.path.basename(video_path))
    try:
        shutil.copy2(video_path, temp_video_path)

        # override agent paths
        if hasattr(visual_agent, "INPUT_VIDEO_DIR"):
            visual_agent.INPUT_VIDEO_DIR = temp_dir
        if hasattr(visual_agent, "log_dir"):
            visual_agent.log_dir = VISUAL_RESULTS_DIR
        if hasattr(visual_agent, "OUTPUT_DIR"):
            visual_agent.OUTPUT_DIR = VISUAL_RESULTS_DIR

        # call visual processing (synchronous main)
        visual_agent.process_videos_and_run_batched()

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temp directory: {e}")


def run_audio_transcription_for_video(video_path: str):
    base = Path(video_path).stem
    audio_tmp = os.path.join(tempfile.gettempdir(), f"temp_audio_for_{base}.wav")

    # override audio agent paths
    if hasattr(audio_agent, "input_folder"):
        audio_agent.input_folder = AUDIO_INPUT_DIR
    if hasattr(audio_agent, "output_folder"):
        audio_agent.output_folder = AUDIO_OUTPUT_DIR

    print(f"  Extracting audio to {audio_tmp}...")
    audio_agent.extract_audio(video_path, audio_tmp, fps=16000)

    print("  Transcribing (this may take a while)...")
    result, model = audio_agent.transcribe_faster_whisper(
        audio_tmp,
        audio_agent.model_size if hasattr(audio_agent, "model_size") else "medium",
        model=None
    )

    print("  Saving outputs...")
    _, json_path = audio_agent.save_outputs(result, video_path, AUDIO_OUTPUT_DIR)

    try:
        if os.path.exists(audio_tmp):
            os.remove(audio_tmp)
    except Exception:
        pass

    return json_path


def run_reasoning_on_video(video_stem: str):
    # override reasoner paths
    if hasattr(reason_agent, "VISUAL_RESULTS_DIR"):
        reason_agent.VISUAL_RESULTS_DIR = VISUAL_RESULTS_DIR
    if hasattr(reason_agent, "AUDIO_RESULTS_DIR"):
        reason_agent.AUDIO_RESULTS_DIR = AUDIO_OUTPUT_DIR
    if hasattr(reason_agent, "OUTPUT_DIR"):
        reason_agent.OUTPUT_DIR = REASON_OUTPUT_DIR

    visual_batches = reason_agent.load_visual_frames(video_stem, VISUAL_RESULTS_DIR)
    audio_json = reason_agent.load_audio_transcript(video_stem, AUDIO_OUTPUT_DIR)

    print(f"  Visual batches loaded: {len(visual_batches)}")
    print(f"  Audio segments loaded: {len(audio_json.get('segments', [])) if audio_json else 0}")

    # Use reason_agent to run reasoning (it will query RAG internally if available)
    result = reason_agent.run_reasoning_agent(
        video_stem,
        visual_batches,
        audio_json,
        fps=getattr(reason_agent, "FPS", 1)
    )

    saved = reason_agent.save_reasoning_result(video_stem, result, REASON_OUTPUT_DIR)
    return saved, result


def main():
    clear_directory(VISUAL_RESULTS_DIR)
    clear_directory(AUDIO_OUTPUT_DIR)
    clear_directory(REASON_OUTPUT_DIR)

    test_video = Path(TEST_VIDEO_PATH)
    if not test_video.exists():
        raise FileNotFoundError(f"Test video not found: {TEST_VIDEO_PATH}")

    print("=" * 70)
    print("MULTIMODAL PIPELINE ORCHESTRATOR (with RAG)")
    print("=" * 70)
    print(f"Test video: {test_video.name}")
    print(f"Visual results → {VISUAL_RESULTS_DIR}")
    print(f"Audio results  → {AUDIO_OUTPUT_DIR}")
    print(f"Final results  → {REASON_OUTPUT_DIR}")
    print("=" * 70)

    print("\nSTEP 1/3: VISUAL ANALYSIS")
    run_visual_agent_for_video(str(test_video))

    print("\nSTEP 2/3: AUDIO TRANSCRIPTION")
    audio_json_path = run_audio_transcription_for_video(str(test_video))
    print(f"✓ Audio transcript saved: {audio_json_path}")

    print("\nSTEP 3/3: MULTIMODAL REASONING")
    video_stem = test_video.stem
    saved_path, result = run_reasoning_on_video(video_stem)

    print("\nPIPELINE COMPLETE")
    print(f"✓ Final reasoning result: {saved_path}")


if __name__ == "__main__":
    main()