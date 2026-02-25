#!/usr/bin/env python3
"""
master.py - Unified pipeline for AR glasses multimodal processing
Usage: python3 master.py test4
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
INPUT_VIDEO_DIR = Path(os.environ.get("INPUT_VIDEO_DIR", "/workspace/qwen/agents/roadshow/test"))
VISUAL_DIR = Path("/workspace/qwen/agents/roadshow/test_result/visual_results")
AUDIO_DIR = Path("/workspace/qwen/agents/roadshow/test_result/transcripts")
REASONING_DIR = Path("/workspace/qwen/agents/roadshow/test_result/reasoning_results")
ACTION_DIR = Path("/workspace/qwen/agents/roadshow/test_result/action_results")

# Agent script paths (adjust if needed)
VISUAL_AGENT = Path("visual_agent.py")
AUDIO_AGENT = Path("audio_agent.py")
REASONING_AGENT = Path("reasoning_agent.py")
INFERENCE_AGENT = Path("inference_agent.py")

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MasterPipeline")

def run_agent(script: Path, video_name: str, stage_name: str):
    """Run an agent script and check for errors."""
    logger.info(f"[{stage_name}] Starting...")
    
    try:
        result = subprocess.run(
            ["python3", str(script), video_name],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"[{stage_name}] ✓ Completed")
        
        # Print stdout if there's useful info
        if result.stdout.strip():
            print(result.stdout)
        
        return True, stage_name
    except subprocess.CalledProcessError as e:
        logger.error(f"[{stage_name}] ✗ Failed")
        logger.error(f"Error output:\n{e.stderr}")
        if e.stdout:
            logger.error(f"Standard output:\n{e.stdout}")
        return False, stage_name

def run_parallel_agents(video_path: str, video_stem: str):
    """Run visual and audio agents in parallel."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        futures = {
            executor.submit(run_agent, VISUAL_AGENT, str(video_path), "VISUAL AGENT"): "visual",
            executor.submit(run_agent, AUDIO_AGENT, video_stem, "AUDIO AGENT"): "audio"
        }
        
        # Wait for completion
        results = {}
        for future in as_completed(futures):
            success, stage_name = future.result()
            results[futures[future]] = success
        
        return all(results.values())

def verify_outputs(video_stem: str):
    """Check if required output files exist."""
    checks = {
        "Visual": list(VISUAL_DIR.glob(f"{video_stem}_*.json")),
        "Audio": AUDIO_DIR / f"{video_stem}_transcript.json",
        "Reasoning": REASONING_DIR / f"{video_stem}_final_reasoning.json",
        "Inference": ACTION_DIR / f"{video_stem}_multi_action.json"
    }
    
    logger.info("\n" + "="*60)
    logger.info("OUTPUT VERIFICATION")
    logger.info("="*60)
    
    all_good = True
    for stage, path in checks.items():
        if isinstance(path, list):
            exists = len(path) > 0
            status = f"✓ {len(path)} files" if exists else "✗ Missing"
        else:
            exists = path.exists()
            status = "✓ Found" if exists else "✗ Missing"
            # Show what we're looking for if missing
            if not exists:
                status += f" (looking for: {path.name})"
        
        logger.info(f"{stage:12} : {status}")
        if not exists:
            all_good = False
    
    # List what actually exists in action_results
    if not all_good:
        logger.info("\nFiles in action_results directory:")
        if ACTION_DIR.exists():
            files = list(ACTION_DIR.glob("*.json"))
            if files:
                for f in files:
                    logger.info(f"  - {f.name}")
            else:
                logger.info("  (empty)")
        else:
            logger.info("  (directory doesn't exist)")
    
    logger.info("="*60 + "\n")
    return all_good

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 master.py <video_name>")
        print("Example: python3 master.py test4")
        sys.exit(1)
    
    video_input = sys.argv[1]
    
    # Remove extension if provided
    video_stem = Path(video_input).stem
    
    # Check if video exists
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_path = None
    
    for ext in video_extensions:
        candidate = INPUT_VIDEO_DIR / f"{video_stem}{ext}"
        if candidate.exists():
            video_path = candidate
            break
    
    if not video_path:
        logger.error(f"Video not found: {video_stem} (checked {INPUT_VIDEO_DIR})")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info(f"PROCESSING: {video_path.name}")
    logger.info("="*60)
    
    # Stage 1 & 2: Visual + Audio Agents (Parallel)
    logger.info("\n[PARALLEL PROCESSING] Running Visual and Audio agents...")
    if not run_parallel_agents(video_path, video_stem):
        logger.error("Pipeline aborted: Visual or Audio agent failed")
        sys.exit(1)
    
    # Stage 3: Reasoning Agent
    success, _ = run_agent(REASONING_AGENT, video_stem, "REASONING AGENT")
    if not success:
        logger.error("Pipeline aborted at Reasoning Agent")
        sys.exit(1)
    
    # Stage 4: Inference Agent
    success, _ = run_agent(INFERENCE_AGENT, video_stem, "INFERENCE AGENT")
    if not success:
        logger.error("Pipeline aborted at Inference Agent")
        sys.exit(1)
    
    # Verification
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETED")
    logger.info("="*60)
    
    if verify_outputs(video_stem):
        logger.info("✓ All outputs generated successfully")
        logger.info(f"\nResults location:")
        logger.info(f"  Visual    : {VISUAL_DIR}")
        logger.info(f"  Audio     : {AUDIO_DIR}")
        logger.info(f"  Reasoning : {REASONING_DIR}")
        logger.info(f"  Inference : {ACTION_DIR}")
    else:
        logger.warning("⚠ Some outputs are missing - check logs above")
    
    logger.info("="*60 + "\n")

if __name__ == "__main__":
    main()