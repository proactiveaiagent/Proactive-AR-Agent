#!/usr/bin/env python3
"""
Real-time streaming server for AR device
- Receives video/audio from AR device
- Processes with visual_agent.py and audio_agent.py
- Runs reasoning_agent.py for multimodal fusion
- Runs inference_agent.py for AR assistance
- Sends solutions back to AR device
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
import json
import wave
import base64
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
import threading
import logging
from typing import Dict, List, Optional
import uuid
import time

app = Flask(__name__)
CORS(app)

# ═════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════
TEMP_DIR = Path("/workspace/qwen/agents/roadshow/streaming_temp")
RESULTS_DIR = Path("/workspace/qwen/agents/roadshow/streaming_results")

# Agent paths
VISUAL_AGENT = Path("visual_agent.py")
AUDIO_AGENT = Path("audio_agent.py")
REASONING_AGENT = Path("reasoning_agent.py")
INFERENCE_AGENT = Path("inference_agent.py")

# Streaming parameters
FRAME_TRIGGER_COUNT = 15  # Process every 15 frames
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1

# Create directories
TEMP_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════
# Session Management
# ═════════════════════════════════════════════════════════
class StreamingSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.frame_buffer = []
        self.audio_buffer = []
        self.frame_count = 0
        self.is_speaking = False
        self.processing = False

        # Processing completion flags
        self.visual_processed = False
        self.audio_processed = False
        self.visual_success = False
        self.audio_success = False

        # Latest results (tagged with run_id + timestamp)
        self.latest_inference = None
        self.latest_inference_time = 0.0
        self.latest_run_id = None

        # Current run metadata (set by /session/start)
        self.current_run_id = None
        self.current_run_start = 0.0

        # Session directories
        self.session_dir = TEMP_DIR / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.created_at = datetime.now()
        logger.info(f"Created session: {session_id}")


sessions: Dict[str, StreamingSession] = {}
sessions_lock = threading.Lock()


def get_or_create_session(session_id: str) -> StreamingSession:
    """Thread-safe session retrieval/creation"""
    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = StreamingSession(session_id)
        return sessions[session_id]


# ═════════════════════════════════════════════════════════
# ENDPOINT 0: Start / Reset Session Run
# ═════════════════════════════════════════════════════════
@app.route('/session/start', methods=['POST'])
def session_start():
    """
    Start a new client run for a session. Returns run_id and start_time.

    Request:
    {
      "session_id": "string",
      "run_id": "optional string"  # if not provided, server generates one
    }

    Response:
    {
      "session_id": "...",
      "run_id": "...",
      "start_time": <epoch_seconds>
    }
    """
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', 'default')
        run_id = data.get('run_id') or str(uuid.uuid4())
        now_ts = time.time()

        session = get_or_create_session(session_id)

        # Reset processing flags for new run
        session.visual_processed = False
        session.audio_processed = False
        session.visual_success = False
        session.audio_success = False
        session.current_run_id = run_id
        session.current_run_start = now_ts

        logger.info(f"[SESSION] Started run {run_id} for session {session_id} at {now_ts}")

        return jsonify({
            "session_id": session_id,
            "run_id": run_id,
            "start_time": now_ts
        })
    except Exception as e:
        logger.exception("Error in session_start")
        return jsonify({"error": str(e)}), 500


# ═════════════════════════════════════════════════════════
# ENDPOINT 1: Health Check
# ═════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health_check():
    """Server health check"""
    agents_exist = all([
        VISUAL_AGENT.exists(),
        AUDIO_AGENT.exists(),
        REASONING_AGENT.exists(),
        INFERENCE_AGENT.exists()
    ])

    return jsonify({
        "status": "running",
        "active_sessions": len(sessions),
        "timestamp": datetime.now().isoformat(),
        "agents_available": agents_exist,
        "endpoints": {
            "video_stream": "/stream/video",
            "audio_stream": "/stream/audio",
            "get_solution": "/solution/get",
            "session_status": "/session/status",
            "session_start": "/session/start",
            "stream_end": "/stream/end"
        }
    })


# ═════════════════════════════════════════════════════════
# ENDPOINT 2: Video Streaming
# ═════════════════════════════════════════════════════════
@app.route('/stream/video', methods=['POST'])
def receive_video_frame():
    """
    Receive JPEG frame from AR device

    Request body (JSON):
    {
        "session_id": "string",
        "frame_data": "base64_encoded_jpeg",
        "timestamp": "ISO timestamp"
    }
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        frame_b64 = data.get('frame_data')
        timestamp = data.get('timestamp', datetime.now().isoformat())

        if not frame_b64:
            return jsonify({'error': 'No frame data provided'}), 400

        session = get_or_create_session(session_id)

        # Decode and save frame
        frame_bytes = base64.b64decode(frame_b64)
        frame_filename = f"frame_{session.frame_count:06d}.jpg"
        frame_path = session.session_dir / frame_filename

        with open(frame_path, 'wb') as f:
            f.write(frame_bytes)

        session.frame_buffer.append({
            'path': str(frame_path),
            'timestamp': timestamp,
            'frame_id': session.frame_count
        })
        session.frame_count += 1

        # Trigger visual processing when buffer reaches threshold
        if len(session.frame_buffer) >= FRAME_TRIGGER_COUNT and not session.visual_processed and not session.processing:
            threading.Thread(
                target=process_visual_only,
                args=(session_id,),
                daemon=True
            ).start()

        return jsonify({
            'status': 'received',
            'session_id': session_id,
            'frame_id': session.frame_count - 1,
            'buffer_size': len(session.frame_buffer),
            'will_process': len(session.frame_buffer) >= FRAME_TRIGGER_COUNT
        })

    except Exception as e:
        logger.error(f"Error in receive_video_frame: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════════════════
# ENDPOINT 3: Audio Streaming
# ═════════════════════════════════════════════════════════
@app.route('/stream/audio', methods=['POST'])
def receive_audio_chunk():
    """
    Receive PCM audio chunk from AR device

    Request body (JSON):
    {
        "session_id": "string",
        "audio_data": "base64_encoded_pcm",
        "is_speaking": bool,
        "timestamp": "ISO timestamp"
    }
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        audio_b64 = data.get('audio_data')
        is_speaking = data.get('is_speaking', False)
        timestamp = data.get('timestamp', datetime.now().isoformat())

        if not audio_b64:
            return jsonify({'error': 'No audio data provided'}), 400

        session = get_or_create_session(session_id)

        # Decode audio
        audio_bytes = base64.b64decode(audio_b64)

        # Update speaking state
        was_speaking = session.is_speaking
        session.is_speaking = is_speaking

        # Add to buffer
        session.audio_buffer.append({
            'data': audio_bytes,
            'timestamp': timestamp,
            'is_speaking': is_speaking
        })

        # If user just stopped speaking and we have audio, process it
        if was_speaking and not is_speaking and len(session.audio_buffer) > 0:
            if not session.audio_processed and not session.processing:
                threading.Thread(
                    target=process_audio_only,
                    args=(session_id,),
                    daemon=True
                ).start()

        return jsonify({
            'status': 'received',
            'session_id': session_id,
            'buffer_size': len(session.audio_buffer),
            'is_speaking': is_speaking
        })

    except Exception as e:
        logger.error(f"Error in receive_audio_chunk: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════════════════
# ENDPOINT 4: Get Latest Solution
# ═════════════════════════════════════════════════════════
@app.route('/solution/get', methods=['POST'])
def get_latest_solution():
    """
    Get latest inference solution for AR display

    Request body (JSON):
    {
        "session_id": "string",
        "last_checked": float (timestamp, optional),
        "run_id": "optional string"  # prefer results for this run
    }
    """
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', 'default')
        last_checked = float(data.get('last_checked', 0) or 0)
        requested_run_id = data.get('run_id')

        if session_id not in sessions:
            return jsonify({
                "status": "no_session",
                "has_solution": False
            })

        session = sessions[session_id]

        # No solution yet
        if session.latest_inference is None:
            return jsonify({
                "status": "no_solution",
                "has_solution": False,
                "processing": session.processing,
                "visual_processed": session.visual_processed,
                "audio_processed": session.audio_processed
            })

        # If a run_id is provided, ensure we only return solutions created for that run.
        if requested_run_id:
            if session.latest_run_id != requested_run_id:
                return jsonify({
                    "status": "no_new_solution_for_run",
                    "has_solution": False,
                    "requested_run_id": requested_run_id,
                    "latest_run_id": session.latest_run_id,
                    "processing": session.processing
                })

        # Check timestamp vs last_checked
        if session.latest_inference_time <= last_checked:
            return jsonify({
                "status": "no_new_solution",
                "has_solution": False,
                "last_solution_time": session.latest_inference_time,
                "processing": session.processing,
                "latest_run_id": session.latest_run_id
            })

        # Return new solution
        logger.info(f"[SOLUTION] Sending solution to AR device for session {session_id} (run {session.latest_run_id})")

        return jsonify({
            "status": "new_solution",
            "has_solution": True,
            "timestamp": session.latest_inference_time,
            "run_id": session.latest_run_id,
            "processing": session.processing,
            "solution": session.latest_inference
        })

    except Exception as e:
        logger.error(f"Error in get_latest_solution: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════════════════
# ENDPOINT 5: Session Status
# ═════════════════════════════════════════════════════════
@app.route('/session/status', methods=['POST'])
def session_status():
    """Get session status"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')

        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404

        session = sessions[session_id]

        return jsonify({
            'session_id': session_id,
            'created_at': session.created_at.isoformat(),
            'frames_received': session.frame_count,
            'frames_buffered': len(session.frame_buffer),
            'audio_chunks_buffered': len(session.audio_buffer),
            'is_speaking': session.is_speaking,
            'processing': session.processing,
            'visual_processed': session.visual_processed,
            'audio_processed': session.audio_processed,
            'visual_success': session.visual_success,
            'audio_success': session.audio_success,
            'has_solution': session.latest_inference is not None,
            'last_solution_time': session.latest_inference_time,
            'latest_run_id': session.latest_run_id,
            'current_run_id': session.current_run_id,
            'current_run_start': session.current_run_start
        })

    except Exception as e:
        logger.error(f"Error in session_status: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════════════════
# ENDPOINT 6: Stream End Signal
# ═════════════════════════════════════════════════════════
@app.route('/stream/end', methods=['POST'])
def receive_end_signal():
    """
    Explicit endpoint for when the video/audio finishes.
    Triggers processing of any remaining data.
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        session = get__or_create_session(session_id)

        logger.info(f"[SIGNAL] Received end signal for session {session_id}")

        # Process any remaining video frames
        if len(session.frame_buffer) > 0 and not session.visual_processed:
            threading.Thread(
                target=process_visual_only,
                args=(session_id,),
                daemon=True
            ).start()

        # Process any remaining audio chunks
        if len(session.audio_buffer) > 0 and not session.audio_processed:
            threading.Thread(
                target=process_audio_only,
                args=(session_id,),
                daemon=True
            ).start()

        return jsonify({
            'status': 'processing_triggered',
            'session_id': session_id,
            'visual_will_process': len(session.frame_buffer) > 0 and not session.visual_processed,
            'audio_will_process': len(session.audio_buffer) > 0 and not session.audio_processed
        })

    except Exception as e:
        logger.error(f"Error in receive_end_signal: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════════════════
# Core Processing Functions
# ═════════════════════════════════════════════════════════
def process_visual_only(session_id: str):
    """Process visual data only"""
    try:
        if session_id not in sessions:
            return

        session = sessions[session_id]

        # Prevent concurrent processing
        if session.processing or session.visual_processed:
            logger.info(f"[VISUAL] Session {session_id} already processing or processed, skipping")
            return

        session.processing = True
        logger.info(f"[VISUAL] Starting visual processing for session {session_id}")

        run_id = session.current_run_id or f"{session_id}_{int(datetime.now().timestamp())}"
        batch_id = f"{run_id}_{int(datetime.now().timestamp())}"

        # Process visual
        visual_success = run_visual_agent(session, batch_id)
        
        session.visual_processed = True
        session.visual_success = visual_success
        session.processing = False

        logger.info(f"[VISUAL] Completed visual processing for session {session_id} (success={visual_success})")

        # Check if we can proceed to reasoning
        check_and_run_reasoning(session_id, batch_id, run_id)

    except Exception as e:
        logger.error(f"[VISUAL] Error in process_visual_only: {e}", exc_info=True)
        if session_id in sessions:
            sessions[session_id].processing = False
            sessions[session_id].visual_processed = True


def process_audio_only(session_id: str):
    """Process audio data only"""
    try:
        if session_id not in sessions:
            return

        session = sessions[session_id]

        # Prevent concurrent processing
        if session.processing or session.audio_processed:
            logger.info(f"[AUDIO] Session {session_id} already processing or processed, skipping")
            return

        session.processing = True
        logger.info(f"[AUDIO] Starting audio processing for session {session_id}")

        run_id = session.current_run_id or f"{session_id}_{int(datetime.now().timestamp())}"
        batch_id = f"{run_id}_{int(datetime.now().timestamp())}"

        # Process audio
        audio_success = run_audio_agent(session, batch_id)
        
        session.audio_processed = True
        session.audio_success = audio_success
        session.processing = False

        logger.info(f"[AUDIO] Completed audio processing for session {session_id} (success={audio_success})")

        # Check if we can proceed to reasoning
        check_and_run_reasoning(session_id, batch_id, run_id)

    except Exception as e:
        logger.error(f"[AUDIO] Error in process_audio_only: {e}", exc_info=True)
        if session_id in sessions:
            sessions[session_id].processing = False
            sessions[session_id].audio_processed = True


def check_and_run_reasoning(session_id: str, batch_id: str, run_id: str):
    """
    Check if both visual and audio are processed, then run reasoning and inference.
    This is the KEY function that enforces the requirement.
    """
    try:
        if session_id not in sessions:
            return

        session = sessions[session_id]

        # CRITICAL CHECK: Both must be processed before reasoning can start
        if not (session.visual_processed and session.audio_processed):
            logger.info(f"[REASONING] Waiting for both modalities. Visual: {session.visual_processed}, Audio: {session.audio_processed}")
            return

        # Check if at least one succeeded
        if not (session.visual_success or session.audio_success):
            logger.error(f"[REASONING] Both visual and audio processing failed for session {session_id}")
            return

        # Prevent concurrent processing
        if session.processing:
            logger.info(f"[REASONING] Session {session_id} already processing, skipping")
            return

        session.processing = True
        logger.info(f"[REASONING] Both modalities ready! Starting reasoning for session {session_id}")

        # Run reasoning agent
        reasoning_success = run_reasoning_agent(session, batch_id)

        if reasoning_success:
            # Run inference agent
            run_inference_agent(session, batch_id, run_id=run_id)

        session.processing = False
        logger.info(f"[PIPELINE] ✓ Completed full pipeline for session {session_id}")

    except Exception as e:
        logger.error(f"[REASONING] Error in check_and_run_reasoning: {e}", exc_info=True)
        if session_id in sessions:
            sessions[session_id].processing = False


# ═════════════════════════════════════════════════════════
# Agent Execution Functions
# ═════════════════════════════════════════════════════════
def run_visual_agent(session: StreamingSession, batch_id: str) -> bool:
    """Run visual_agent.py on buffered frames"""
    try:
        logger.info(f"[VISUAL] Processing {len(session.frame_buffer)} frames")

        # Get frames and clear buffer
        with sessions_lock:
            frames = session.frame_buffer.copy()
            session.frame_buffer.clear()

        if len(frames) == 0:
            return False

        # Create temporary video from frames
        video_path = session.session_dir / f"{batch_id}.mp4"
        create_video_from_frames([f['path'] for f in frames], video_path)

        # Run visual agent
        result = subprocess.run(
            ["python3", str(VISUAL_AGENT), str(video_path)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            logger.info(f"[VISUAL] ✓ Success")
            return True
        else:
            logger.error(f"[VISUAL] ✗ Failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"[VISUAL] Error: {e}")
        return False


def run_audio_agent(session: StreamingSession, batch_id: str) -> bool:
    """
    Run audio_agent.py - Final Fix.
    Creates a valid MP4 with a dummy black video stream to satisfy 
    MoviePy's 'first frame' requirement while keeping Whisper audio intact.
    """
    try:
        logger.info(f"[AUDIO] Processing {len(session.audio_buffer)} chunks")

        with sessions_lock:
            audio_chunks = session.audio_buffer.copy()
            session.audio_buffer.clear()

        if len(audio_chunks) == 0:
            return False

        # 1. Define paths - Must save to the directory the agent scans
        input_dir = Path("/workspace/qwen/agents/roadshow/test")
        input_dir.mkdir(parents=True, exist_ok=True)
        
        temp_wav = input_dir / f"{batch_id}_temp.wav"
        final_video_path = input_dir / f"{batch_id}.mp4"

        # 2. Write raw chunks to a temporary WAV file
        with wave.open(str(temp_wav), 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            for chunk in audio_chunks:
                wav_file.writeframes(chunk['data'])

        # 3. Create a COMPATIBLE MP4 with a synthetic video track
        logger.info(f"[AUDIO] Muxing audio into compatible MP4 for agent...")
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=640x480:r=10",
            "-i", str(temp_wav),
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(final_video_path)
        ], capture_output=True, check=True)

        # 4. Set up environment variables for the agent
        env = os.environ.copy()
        env["INPUT_VIDEO_DIR"] = str(input_dir)
        env["AUDIO_RESULTS_DIR"] = "/workspace/qwen/agents/roadshow/test_result/transcripts"

        # 5. Call Agent: Pass batch_id
        logger.info(f"[AUDIO] Executing agent for batch: {batch_id}")
        result = subprocess.run(
            ["python3", str(AUDIO_AGENT), batch_id],
            env=env,
            capture_output=True,
            text=True,
            timeout=120
        )

        # Cleanup: Remove intermediate files
        if temp_wav.exists():
            temp_wav.unlink()

        # Final check based on Agent's output format
        if result.returncode == 0 and "Processed 1 videos" in result.stdout:
            logger.info(f"[AUDIO] ✓ Success: {batch_id}")
            return True
        else:
            logger.error(f"[AUDIO] ✗ Agent processed 0 videos or failed.")
            logger.error(f"Agent Stdout: {result.stdout}")
            return False

    except Exception as e:
        logger.error(f"[AUDIO] Pipeline Error: {e}")
        return False


def run_reasoning_agent(session: StreamingSession, batch_id: str) -> bool:
    """Run reasoning_agent.py to fuse visual + audio"""
    try:
        logger.info(f"[REASONING] Fusing multimodal data")

        # Run reasoning agent
        result = subprocess.run(
            ["python3", str(REASONING_AGENT), batch_id],
            capture_output=True,
            text=True,
            timeout=150
        )

        if result.returncode == 0:
            logger.info(f"[REASONING] ✓ Success")
            return True
        else:
            logger.error(f"[REASONING] ✗ Failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"[REASONING] Error: {e}")
        return False


def run_inference_agent(session: StreamingSession, batch_id: str, run_id: Optional[str] = None) -> bool:
    """Run inference_agent.py to generate AR solution"""
    try:
        logger.info(f"[INFERENCE] Generating AR solution")

        # Run inference agent
        result = subprocess.run(
            ["python3", str(INFERENCE_AGENT), batch_id],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            logger.info(f"[INFERENCE] ✓ Success")

            # Load the inference result
            inference_file = Path("/workspace/qwen/agents/roadshow/test_result/action_results") / f"{batch_id}_multi_action.json"

            if inference_file.exists():
                with open(inference_file, 'r', encoding='utf-8') as f:
                    inference_data = json.load(f)

                # Store in session and tag with run_id + timestamp
                session.latest_inference = format_inference_for_ar(inference_data)
                ts = time.time()
                session.latest_inference_time = ts
                session.latest_run_id = run_id or session.current_run_id or batch_id

                logger.info(f"[INFERENCE] ✓ Solution ready for AR device (run_id={session.latest_run_id}, ts={ts})")
                return True
            else:
                logger.error(f"[INFERENCE] ✗ Output file not found: {inference_file}")
                return False
        else:
            logger.error(f"[INFERENCE] ✗ Failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"[INFERENCE] Error: {e}")
        return False


# ═════════════════════════════════════════════════════════
# Helper Functions
# ═════════════════════════════════════════════════════════
def create_video_from_frames(frame_paths: List[str], output_path: Path):
    """Create MP4 video from frame images"""
    try:
        if len(frame_paths) == 0:
            return

        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            logger.error(f"Could not read first frame: {frame_paths[0]}")
            return

        height, width, _ = first_frame.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10  # 10 fps for streaming
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Write frames
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)

        out.release()
        logger.info(f"Created video: {output_path} with {len(frame_paths)} frames")

    except Exception as e:
        logger.error(f"Error creating video: {e}")


def format_inference_for_ar(inference_dict) -> Dict:
    """
    Format inference agent output for AR device display

    Input: Raw output from inference_agent.py
    Output: Structured data for AR glasses
    """
    try:
        # Use provided argument name and provide safe defaults
        primary_assessment = inference_dict.get('primary_assessment', '')
        assistance_suite = inference_dict.get('assistance_suite', [])

        # Format each assistance item
        formatted_items = []
        for item in assistance_suite:
            formatted_items.append({
                'category': item.get('need_category', 'Unknown'),
                'need': item.get('inferred_need', ''),
                'hud_message': item.get('ar_solution', ''),
                'action_tip': item.get('action_detail', '')
            })

        return {
            'summary': primary_assessment,
            'assistance_items': formatted_items,
            'timestamp': datetime.now().isoformat(),

            # AR Display Structure
            'display': {
                'main_message': formatted_items[0]['hud_message'] if len(formatted_items) > 0 else '',
                'secondary_messages': [item['hud_message'] for item in formatted_items[1:]],
                'action_tips': [item['action_tip'] for item in formatted_items]
            },

            # Quick access by category
            'by_category': {
                item['category']: {
                    'need': item['need'],
                    'solution': item['hud_message'],
                    'tip': item['action_tip']
                }
                for item in formatted_items
            }
        }

    except Exception as e:
        logger.exception(f"Error formatting inference {e}")
        return {
            'summary': 'Error processing solution',
            'assistance_items': [],
            'display': {'main_message': 'Processing error', 'secondary_messages': [], 'action_tips': []},
            'by_category': {}
        }


# ═════════════════════════════════════════════════════════
# Server Startup
# ═════════════════════════════════════════════════════════
def startup_server():
    """Initialize server components"""
    logger.info("=" * 70)
    logger.info("STARTING AR STREAMING SERVER")
    logger.info("=" * 70)

    # Check if agents exist
    logger.info("\nChecking agent availability...")
    agents = {
        'Visual Agent': VISUAL_AGENT,
        'Audio Agent': AUDIO_AGENT,
        'Reasoning Agent': REASONING_AGENT,
        'Inference Agent': INFERENCE_AGENT
    }

    all_available = True
    for name, path in agents.items():
        exists = path.exists()
        status = '✓' if exists else '✗'
        logger.info(f"  {status} {name}: {path}")
        if not exists:
            all_available = False

    if all_available:
        logger.info("\n" + "=" * 70)
        logger.info("✓ SERVER READY")
        logger.info("=" * 70)
        logger.info("\nProcessing Configuration:")
        logger.info(f"  • Frames per batch: {FRAME_TRIGGER_COUNT}")
        logger.info(f"  • Audio sample rate: {SAMPLE_RATE} Hz")
        logger.info(f"  • Temp directory: {TEMP_DIR}")
        logger.info("\nProcessing Logic:")
        logger.info("  • Visual and Audio agents run INDEPENDENTLY")
        logger.info("  • Reasoning agent starts ONLY after BOTH complete")
        logger.info("  • Use /stream/end to force processing of remaining data")
        logger.info("\nAvailable endpoints:")
        logger.info("  POST /session/start       - Start a new client run (returns run_id)")
        logger.info("  GET  /health              - Health check")
        logger.info("  POST /stream/video        - Receive video frames")
        logger.info("  POST /stream/audio        - Receive audio chunks")
        logger.info("  POST /stream/end          - Signal end of stream (process remaining data)")
        logger.info("  POST /solution/get        - Get latest AR solution (accepts run_id + last_checked)")
        logger.info("  POST /session/status      - Get session status")
        logger.info("=" * 70)
    else:
        logger.error("\n✗ SERVER STARTUP WARNING - Some agents are missing")
        logger.error("  Server will run but processing may fail")
        logger.error("=" * 70)


# ═════════════════════════════════════════════════════════
# Main Entry Point
# ═════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Run startup initialization
    startup_server()

    # Start Flask server
    logger.info("\nStarting Flask server on 0.0.0.0:5000...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )