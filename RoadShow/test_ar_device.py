#!/usr/bin/env python3
"""
AR Device Simulator for Testing Streaming Server
Supports videos with or without audio - agents activate after video ends
This modified version starts a run via /session/start and uses run_id + start_time
when polling /solution/get so it won't return results from previous runs.
"""

import requests
import base64
import time
import json
from pathlib import Path
import cv2
import numpy as np
import subprocess

# Server configuration
SERVER_URL = "http://localhost:5000"
SESSION_ID = "test_user_001"

def test_health_check():
    """Test 1: Check if server is running"""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def start_session_run():
    """Call server to start a new run and get run_id + start_time"""
    print("\nStarting session run...")
    try:
        resp = requests.post(f"{SERVER_URL}/session/start", json={"session_id": SESSION_ID}, timeout=5)
        data = resp.json()
        run_id = data.get("run_id")
        start_time = float(data.get("start_time", time.time()))
        print(f"Started run: run_id={run_id}, start_time={start_time}")
        return run_id, start_time
    except Exception as e:
        print(f"❌ session start failed: {e}")
        return None, time.time()

def check_video_has_audio(video_path):
    """Check if video has audio track"""
    result = subprocess.run([
        "ffprobe", "-i", str(video_path),
        "-show_streams", "-select_streams", "a",
        "-loglevel", "error"
    ], capture_output=True, text=True)
    has_audio = len(result.stdout.strip()) > 0
    return has_audio

def test_send_video_frames(max_frames=10, run_id=None):
    """Test 2: Send video frames (exactly 8 frames)"""
    print("\n" + "="*70)
    print(f"TEST 2: Send Video Frames (limit: {max_frames} frames)")
    print("="*70)
    test_video = Path("/workspace/qwen/agents/roadshow/test/test4.mp4")
    if not test_video.exists():
        print(f"❌ Test video not found: {test_video}")
        return False, False
    has_audio = check_video_has_audio(test_video)
    print(f"Video has audio: {has_audio}")
    cap = cv2.VideoCapture(str(test_video))
    frame_count = 0
    print(f"Reading video: {test_video}")
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠ Video ended at frame {frame_count}")
            break
        _, buffer = cv2.imencode('.jpg', frame)
        base64_frame = base64.b64encode(buffer).decode('utf-8')
        payload = {
            "session_id": SESSION_ID,
            "frame_data": base64_frame,
            "timestamp": time.time()
        }
        if run_id:
            payload["run_id"] = run_id
        try:
            response = requests.post(
                f"{SERVER_URL}/stream/video",
                json=payload,
                timeout=10
            )
            resp_json = response.json()
            print(f"Frame {frame_count + 1}/{max_frames}: buffer_size={resp_json.get('buffer_size', 0)}")
            frame_count += 1
            time.sleep(0.1)  # Simulate 10fps
        except Exception as e:
            print(f"❌ Frame {frame_count + 1} failed: {e}")
            cap.release()
            return False, has_audio
    cap.release()
    print(f"✓ Sent {frame_count} frames")
    return True, has_audio

def test_send_audio(run_id=None):
    """Test 3: Send audio from test video"""
    print("\n" + "="*70)
    print("TEST 3: Send Audio Chunks")
    print("="*70)
    test_video = Path("/workspace/qwen/agents/roadshow/test/test4.mp4")
    if not test_video.exists():
        print(f"❌ Test video not found: {test_video}")
        return False
    temp_audio = "/tmp/test_audio.wav"
    result = subprocess.run([
        "ffmpeg", "-i", str(test_video),
        "-ar", "16000",
        "-ac", "1",
        "-f", "s16le",
        "-y",
        temp_audio
    ], capture_output=True)
    if result.returncode != 0:
        print(f"❌ Audio extraction failed (video may have no audio)")
        print(f"   Error: {result.stderr.decode()}")
        return False
    try:
        with open(temp_audio, 'rb') as f:
            audio_data = f.read()
    except:
        print(f"❌ Cannot read audio file")
        return False
    if len(audio_data) == 0:
        print(f"❌ Audio file is empty")
        return False
    chunk_size = 16000 * 2
    chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
    print(f"Sending {len(chunks)} audio chunks...")
    for i, chunk in enumerate(chunks):
        base64_audio = base64.b64encode(chunk).decode('utf-8')
        is_speaking = (i < len(chunks) - 1)
        payload = {
            "session_id": SESSION_ID,
            "audio_data": base64_audio,
            "is_speaking": is_speaking,
            "timestamp": time.time()
        }
        if run_id:
            payload["run_id"] = run_id
        try:
            response = requests.post(
                f"{SERVER_URL}/stream/audio",
                json=payload,
                timeout=10
            )
            status = "speaking..." if is_speaking else "stopped ✓ AGENTS TRIGGERED"
            print(f"Chunk {i + 1}/{len(chunks)}: {status}")
            time.sleep(0.5)  # Simulate real-time streaming
        except Exception as e:
            print(f"❌ Chunk {i + 1} failed: {e}")
            return False
    print(f"✓ Sent {len(chunks)} audio chunks")
    return True

def send_end_signal(run_id=None):
    """Send explicit end signal to trigger agents when no audio"""
    print("\n" + "="*70)
    print("SENDING END SIGNAL (No Audio Mode)")
    print("="*70)
    payload = {
        "session_id": SESSION_ID,
        "timestamp": time.time()
    }
    if run_id:
        payload["run_id"] = run_id
    try:
        response = requests.post(
            f"{SERVER_URL}/stream/end",
            json=payload,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        resp_json = response.json()
        print(f"Response: {resp_json}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ End signal failed: {e}")
        return False

def test_poll_solution(run_id=None, last_checked=0):
    """Test 4: Poll for solution (uses run_id + last_checked)"""
    print("\n" + "="*70)
    print("TEST 4: Poll for Solution")
    print("="*70)
    max_polls = 36  # 3 minutes
    poll_interval = 5.0
    print(f"Polling for solution (timeout: 3 minutes, interval: {poll_interval}s)...")
    for i in range(max_polls):
        try:
            payload = {"session_id": SESSION_ID, "last_checked": last_checked}
            if run_id:
                payload["run_id"] = run_id
            response = requests.post(
                f"{SERVER_URL}/solution/get",
                json=payload,
                timeout=5
            )
            data = response.json()
            if data.get("has_solution") and data.get("status") == "new_solution":
                print("\n" + "🎉 " + "="*66)
                print("SOLUTION RECEIVED!")
                print("="*70)
                print(json.dumps(data["solution"], indent=2, ensure_ascii=False))
                print("="*70)
                return True
            elif data.get("processing"):
                elapsed = (i + 1) * poll_interval
                print(f"Poll {i + 1}/{max_polls} ({int(elapsed)}s elapsed): Processing... (waiting)")
            else:
                elapsed = (i + 1) * poll_interval
                # Show helpful debug when server says no_new_solution_for_run
                if data.get("status") == "no_new_solution_for_run":
                    print(f"Poll {i + 1}/{max_polls}: No solution for run_id={run_id} yet (server latest_run_id={data.get('latest_run_id')})")
                else:
                    print(f"Poll {i + 1}/{max_polls} ({int(elapsed)}s elapsed): No new solution yet")
            time.sleep(poll_interval)
        except Exception as e:
            print(f"❌ Poll {i + 1} failed: {e}")
            return False
    print(f"❌ Timeout: No solution received after 3 minutes ({max_polls} polls)")
    return False

def test_session_status():
    """Test 5: Check session status"""
    print("\n" + "="*70)
    print("TEST 5: Session Status")
    print("="*70)
    try:
        response = requests.post(
            f"{SERVER_URL}/session/status",
            json={"session_id": SESSION_ID},
            timeout=5
        )
        print(f"Status: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def run_full_test():
    """Run complete test suite"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  AR DEVICE SIMULATOR - FULL TEST SUITE".center(68) + "█")
    print("█" + "  (8 frames, audio optional)".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    results = []
    # Test 1: Health check
    results.append(("Health Check", test_health_check()))
    if not results[0][1]:
        print("\n❌ Server not running. Please start streaming_server.py first.")
        return
    # Start a session run (get run_id + start_time)
    run_id, run_start = start_session_run()
    # Test 2: Send video frames (8 frames max)
    video_success, has_audio = test_send_video_frames(max_frames=8, run_id=run_id)
    results.append(("Send Video Frames", video_success))
    # Test 3: Send audio OR end signal
    if has_audio:
        print("\n📢 Video has audio - sending audio chunks...")
        audio_success = test_send_audio(run_id=run_id)
        results.append(("Send Audio", audio_success))
    else:
        print("\n🔇 Video has no audio - sending end signal...")
        end_success = send_end_signal(run_id=run_id)
        results.append(("Send End Signal", end_success))
    # Test 4: Poll for solution (use run_id + run_start)
    results.append(("Poll Solution", test_poll_solution(run_id=run_id, last_checked=run_start)))
    # Test 5: Session status
    results.append(("Session Status", test_session_status()))
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print("="*70)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*70)

if __name__ == "__main__":
    run_full_test()