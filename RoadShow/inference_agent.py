import os
import sys
import json
import logging
import re
from pathlib import Path
from openai import OpenAI
from typing import Dict, Any, List

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ActionAgent")

# --- Configuration ---
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:25000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "/models/Qwen2.5-VL-7B-Instruct")
API_KEY = "EMPTY"

REASONING_DIR = Path("/workspace/qwen/agents/roadshow/test_result/reasoning_results")
ACTION_DIR = Path("/workspace/qwen/agents/roadshow/test_result/action_results")
ACTION_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(base_url=OPENAI_API_BASE, api_key=API_KEY)

def generate_multi_help(video_stem: str, reasoning_data: Dict[str, Any]):
    """Analyzes the situation to provide 3 distinct layers of user assistance."""
    
    prompt = f"""
    TASK: Act as an 'AR Multi-Layer Assistant'for chinese user. Analyze the reasoning data and identify exactly 3 distinct user needs and solutions.
    
    [SITUATION ANALYSIS]
    {json.dumps(reasoning_data, indent=2, ensure_ascii=False)}
    
    [INSTRUCTION]
    Identify 3 needs from these categories: 
    1. **Social Etiquette** (What to say/do for cultural correctness)
    2. **Environmental Support** (Safety, navigation, or object interaction)
    3. **Cognitive/Memory** (Past facts, names, or tasks to complete)
    
    [OUTPUT SCHEMA (Strict JSON)]
    {{
      "primary_assessment": "General summary of the situation",
      "assistance_suite": [
        {{
          "need_category": "Social",
          "inferred_need": "Greeting Grandma correctly",
          "ar_solution": "Display: 'Wish Grandma Happy New Year and good health'",
          "action_detail": "Mention the specific greeting: 祝您身体健康，万事如意"
        }},
        {{
          "need_category": "Environment",
          "inferred_need": "Slippery ground",
          "ar_solution": "Alert: 'Careful! Icy steps detected near the vegetable plot'",
          "action_detail": "Remind user to help Grandma walk if they are outside"
        }},
        {{
          "need_category": "Memory/Task",
          "inferred_need": "Relationship Continuity",
          "ar_solution": "Recall: 'Grandma loves the tea you brought last February'",
          "action_detail": "Ask: 'Did you get a chance to try that Biluochun tea yet?'"
        }}
      ]
    }}
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    raw_content = response.choices[0].message.content
    try:
        json_match = re.search(r'(\{.*\})', raw_content, re.DOTALL)
        return json.loads(json_match.group(1)) if json_match else json.loads(raw_content)
    except Exception as e:
        logger.error(f"Failed to generate multi-help: {e}")
        return {"error": "Inference failed"}

def process_action(video_stem: str):
    logger.info(f"Generating Multi-Layer Support for {video_stem}...")
    
    reasoning_file = REASONING_DIR / f"{video_stem}_final_reasoning.json"
    if not reasoning_file.exists():
        logger.error(f"Reasoning file missing.")
        return

    with open(reasoning_file, 'r', encoding='utf-8') as f:
        reasoning_data = json.load(f)

    # Inference
    multi_action_output = generate_multi_help(video_stem, reasoning_data)

    # Save
    out_file = ACTION_DIR / f"{video_stem}_multi_action.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(multi_action_output, f, indent=2, ensure_ascii=False)
    
    # Visual Terminal Output for Testing
    print(f"\n{'='*20} AR ASSISTANT SUITE: {video_stem} {'='*20}")
    for item in multi_action_output.get("assistance_suite", []):
        print(f"▶ [{item['need_category'].upper()}] {item['inferred_need']}")
        print(f"  HUD: {item['ar_solution']}")
        print(f"  TIP: {item['action_detail']}\n")
    print('='*65)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_action(sys.argv[1])