import os
import sys
import base64
import json
import re
import cv2
import gc
from pathlib import Path
from openai import OpenAI

# ---- Configuration ----
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:25000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "/models/Qwen2.5-VL-7B-Instruct")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

INPUT_VIDEO_DIR = os.environ.get("INPUT_VIDEO_DIR", "/workspace/qwen/agents/roadshow/test")
OUTPUT_DIR = os.environ.get("VISUAL_RESULTS_DIR", "/workspace/qwen/agents/roadshow/test_result/visual_results")
FRAME_SAMPLE_FPS = float(os.environ.get("FRAME_SAMPLE_FPS", "1.0"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5")) # Reduced to 5 for VRAM safety

client = OpenAI(base_url=OPENAI_API_BASE, api_key=API_KEY)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_frame_to_base64(frame):
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success: return None
    return base64.b64encode(buffer).decode("utf-8")

def extract_frames_generator(video_path, sample_fps=1.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(video_fps / sample_fps)))
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % interval == 0:
            b64_str = encode_frame_to_base64(frame)
            if b64_str: yield b64_str
        count += 1
    cap.release()

def build_visual_prompt():
    """Enhanced prompt for rich situational evidence."""
    return """Task: Act as an AR visual sensory module. Describe the first-person(chinese) view strictly in JSON for a reasoning agent.

FOCUS AREAS FOR INFERENCE:
1. WHERE (Place): Look for signs, logos, architecture, or furniture that specify the location (e.g., 'Starbucks interior', 'Corporate boardroom', 'Public bus').
2. WHEN (Time): Look for clocks, watches, sun position, or lighting types (artificial vs natural).
3. WHO (Persons): Assign a descriptive 'label_id' (e.g., 'person_waiter_white_shirt'). Describe their attire (formal/casual), tools they carry, and their social distance from the user.
4. WHAT (Activity): Describe the user's hand movements and object interactions.

JSON SCHEMA:
{
  "where": {
    "environment": "e.g., medical clinic, coffee shop",
    "setting": "indoor|outdoor|transit",
    "visual_evidence": "Specific clues like 'menu on wall' or 'medical equipment'"
  },
  "when": {
    "time_clues": "e.g., digital clock showing 14:00, or golden hour sunlight",
    "social_timing": "e.g., middle of a work day, late night"
  },
  "who": [
    {
      "label_id": "unique_trait_based_id",
      "attire_description": "e.g., professional uniform, high-vis vest",
      "interaction": "e.g., handing a paper to user, avoiding eye contact",
      "posture_and_distance": "e.g., standing 1.5m away, leaning forward",
      "relationship_clue": "e.g., wearing an ID badge, carrying a tray"
    }
  ],
  "what": {
    "user_primary_action": "e.g., typing on laptop, holding a coffee cup",
    "salient_objects": [{"label_id": "string", "state": "e.g., open, charging, held by person_A"}]
  },
  "situational_summary": "One sentence factual summary of the scene."
}

RULES:
- NO markdown fences.
- Use literal descriptions (No "I think", just "Visible: [item]").
- Maintain label_ids across the frame batch."""

def clean_json_output(text):
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
    return {"error": "parse_failure", "raw": text[:500]}

def process_video(vid_path):
    video_name = vid_path.stem
    frames_gen = extract_frames_generator(vid_path, sample_fps=FRAME_SAMPLE_FPS)
    batch, batch_idx = [], 0
    for frame_b64 in frames_gen:
        batch.append(frame_b64)
        if len(batch) >= BATCH_SIZE:
            process_batch(batch, video_name, batch_idx)
            batch, batch_idx = [], batch_idx + 1
    if batch: process_batch(batch, video_name, batch_idx)

def process_batch(batch_frames, video_name, batch_idx):
    print(f"  - Analyzing batch {batch_idx} ({len(batch_frames)} frames)...")
    content = [{"type": "text", "text": build_visual_prompt()}]
    for b64 in batch_frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    
    content.append({"type": "text", "text": "Describe the sequence. Keep labels consistent for the same people/objects."})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=2048
        )
        
        # Handle both string and list content
        raw_content = response.choices[0].message.content
        if isinstance(raw_content, list):
            # Extract text from list of content blocks
            text_content = ""
            for item in raw_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content += item.get("text", "")
                elif isinstance(item, str):
                    text_content += item
            raw_content = text_content
        
        parsed_data = clean_json_output(raw_content)
        
        # Ensure parsed_data is a dict
        if not isinstance(parsed_data, dict):
            parsed_data = {"frames": parsed_data if isinstance(parsed_data, list) else [parsed_data]}
        
        parsed_data["_meta"] = {"video": video_name, "batch": batch_idx}
        
        out_path = Path(OUTPUT_DIR) / f"{video_name}_{batch_idx:04d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=2)
        print(f"    ✓ Saved to {out_path.name}")
    except Exception as e:
        print(f"    ✗ Error in batch {batch_idx}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        
        # If target doesn't exist, try looking in INPUT_VIDEO_DIR
        if not target.exists():
            target = Path(INPUT_VIDEO_DIR) / sys.argv[1]
        
        if target.is_file():
            process_video(target)
        elif target.is_dir():
            for v in [f for f in target.glob("**/*") if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]:
                process_video(v)
        else:
            print(f"Error: '{sys.argv[1]}' not found in current directory or {INPUT_VIDEO_DIR}")
    else:
        for v in [f for f in Path(INPUT_VIDEO_DIR).glob("**/*") if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]:
            process_video(v)