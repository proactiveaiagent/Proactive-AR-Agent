import os
import sys
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple

# Import memory functions from your memory_store.py
from memory_store import upsert_memory, find_by_text

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ReasoningAgent")

# --- Configuration ---
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:25000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "/models/Qwen2.5-VL-7B-Instruct")
API_KEY = "EMPTY"

VISUAL_DIR = Path("/workspace/qwen/agents/roadshow/test_result/visual_results")
AUDIO_DIR = Path("/workspace/qwen/agents/roadshow/test_result/transcripts")
REASONING_DIR = Path("/workspace/qwen/agents/roadshow/test_result/reasoning_results")

REASONING_DIR.mkdir(parents=True, exist_ok=True)
client = OpenAI(base_url=OPENAI_API_BASE, api_key=API_KEY)

# ============================================================================
# UTILITIES & PRE-ANALYSIS
# ============================================================================

def load_multimodal_data(video_stem: str) -> Tuple[List[Dict], Dict]:
    visual_files = sorted(VISUAL_DIR.glob(f"{video_stem}_????.json"))
    audio_file = AUDIO_DIR / f"{video_stem}_transcript.json"

    visual_data = []
    for vf in visual_files:
        try:
            with open(vf, 'r', encoding='utf-8') as f:
                visual_data.append(json.load(f))
        except Exception as e:
            logger.error(f"Error loading visual file {vf}: {e}")

    audio_data = {}
    if audio_file.exists():
        try:
            with open(audio_file, 'r', encoding='utf-8') as f:
                audio_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading audio file {audio_file}: {e}")

    return visual_data, audio_data

def detect_festive_context(transcript: str) -> Dict[str, Any]:
    """Detect festive/seasonal context from audio transcript"""
    festive_markers = {
        "春节": ["过年", "新年", "春节", "拜年", "红包", "年好"],
        "中秋": ["中秋", "月饼", "赏月"],
        "端午": ["端午", "粽子"],
        "元宵": ["元宵", "灯笼"],
        "Christmas": ["Christmas", "Merry", "Santa"],
        "New Year": ["Happy New Year", "New Year"]
    }
    
    detected = {"festival": None, "markers": []}
    
    for festival, markers in festive_markers.items():
        for marker in markers:
            if marker in transcript:
                detected["festival"] = festival
                detected["markers"].append(marker)
    
    return detected

def detect_kinship_terms(transcript: str) -> List[str]:
    """Extract kinship and relationship terms from transcript"""
    kinship_patterns = [
        r"(姥姥|奶奶|爷爷|外婆|外公)",  # Grandparents
        r"(妈妈|妈|母亲|娘)",  # Mother
        r"(爸爸|爸|父亲)",  # Father
        r"(阿姨|叔叔|舅舅|姑姑)",  # Aunts/Uncles
        r"(哥哥|姐姐|弟弟|妹妹)",  # Siblings
    ]
    
    found_terms = []
    for pattern in kinship_patterns:
        matches = re.findall(pattern, transcript)
        found_terms.extend(matches)
    
    return list(set(found_terms))

def extract_dynamic_clues(visual: List[Dict], audio: Dict) -> Tuple[List[str], Dict[str, Any]]:
    """Extract clues and context information"""
    clues = []
    transcript = audio.get("transcripts", "")
    
    # Detect festive context
    festive_context = detect_festive_context(transcript)
    
    # Detect kinship terms
    kinship_terms = detect_kinship_terms(transcript)
    
    # Extract all entities from transcript
    found_entities = re.findall(r"[\u4e00-\u9fff]+|[A-Z][a-z]+", transcript)
    clues.extend(found_entities)
    
    # Add kinship terms as priority clues
    clues.extend(kinship_terms)

    # Visual Objects & Context
    has_person = False
    visual_context = {"objects": [], "scene_type": []}
    
    for frame in visual:
        visual_str = str(frame).lower()
        for obj in frame.get("objects", []):
            label = obj.get("label", "").lower()
            visual_context["objects"].append(label)
            if label not in clues:
                clues.append(label)
            if any(x in label for x in ["person", "man", "woman"]):
                has_person = True
        
        # Detect scene context
        if any(x in visual_str for x in ["lantern", "red decoration", "traditional"]):
            visual_context["scene_type"].append("festive")
        if any(x in visual_str for x in ["home", "living room", "kitchen"]):
            visual_context["scene_type"].append("domestic")

    if has_person and not any(k in clues for k in ["person", "man", "woman"]):
        clues.append("person")
    
    context_info = {
        "festive": festive_context,
        "kinship_terms": kinship_terms,
        "visual_context": visual_context,
        "has_person": has_person
    }

    return list(set(clues)), context_info

def enhanced_memory_retrieval(clues: List[str], context_info: Dict[str, Any]) -> List[Dict]:
    """Enhanced memory retrieval with context-aware prioritization"""
    memory_matches = []
    
    # Priority 1: If festive context + kinship terms detected, search for family members
    if context_info["festive"]["festival"] and context_info["kinship_terms"]:
        logger.info(f"Festive context detected: {context_info['festive']['festival']}")
        logger.info(f"Kinship terms found: {context_info['kinship_terms']}")
        
        # Search for each kinship term in memory
        for term in context_info["kinship_terms"]:
            hits = find_by_text(term, mem_type="entity", top_k=3)
            if hits:
                logger.info(f"Found memory matches for kinship term '{term}': {len(hits)} results")
                memory_matches.extend(hits)
        
        # Also search for festival-related memories
        if context_info["festive"]["festival"]:
            festival_hits = find_by_text(context_info["festive"]["festival"], mem_type="episodic", top_k=3)
            if festival_hits:
                memory_matches.extend(festival_hits)
    
    # Priority 2: Search with general clues
    if clues:
        logger.info(f"Searching memory with general clues: {clues[:10]}")  # Limit log output
        for clue in clues:
            hits = find_by_text(clue, mem_type="entity", top_k=2)
            if hits: 
                memory_matches.extend(hits)
            ep_hits = find_by_text(clue, mem_type="episodic", top_k=1)
            if ep_hits: 
                memory_matches.extend(ep_hits)
    
    # Priority 3: If person detected but no specific matches, get recent person entities
    if context_info["has_person"] and not memory_matches:
        logger.info("Person detected but no specific matches. Retrieving recent person entities.")
        memory_matches.extend(find_by_text("person", mem_type="entity", top_k=5))
    
    # Remove duplicates while preserving order
    unique_memories = list({m['id']: m for m in memory_matches}.values())
    
    logger.info(f"Total unique memory entries retrieved: {len(unique_memories)}")
    return unique_memories

# ============================================================================
# CORE REASONING ENGINE (INTELLIGENT INFERENCE)
# ============================================================================

def run_fused_reasoning(video_stem: str, visual: List[Dict], audio: Dict, memory: List[Dict], context_info: Dict[str, Any]):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build context hints for the model
    context_hints = ""
    if context_info["festive"]["festival"]:
        context_hints += f"\n[CONTEXTUAL HINT: Festival Context Detected]\n"
        context_hints += f"Festival: {context_info['festive']['festival']}\n"
        context_hints += f"Festive Markers in Audio: {', '.join(context_info['festive']['markers'])}\n"
    
    if context_info["kinship_terms"]:
        context_hints += f"\n[CONTEXTUAL HINT: Kinship Terms Detected]\n"
        context_hints += f"Kinship Terms in Audio: {', '.join(context_info['kinship_terms'])}\n"
        context_hints += f"IMPORTANT: These kinship terms likely refer to people in the visual scene. Cross-reference with memory entries to identify specific individuals.\n"
    
    prompt = f"""
    TASK: You are the 'Cognitive Core' of AR glasses for chinese user. Perform intelligent multimodal reasoning.
    Your goal is to bridge visual evidence, audio transcripts, and existing memories to understand the scene.

    [SYSTEM CONTEXT]
    Current Time: {current_time}
    {context_hints}

    [INPUT 1: VISUAL DATA]
    {json.dumps(visual, indent=1, ensure_ascii=False)}

    [INPUT 2: AUDIO TRANSCRIPT]
    {json.dumps(audio, indent=1, ensure_ascii=False)}

    [INPUT 3: CANDIDATE MEMORIES]
    {json.dumps(memory, indent=1, ensure_ascii=False) if memory else "No relevant memories found."}

    [REASONING PROTOCOL]
    1. **Priority Identity Resolution (CRITICAL)**:
       - **Festive + Kinship Context**: When festive markers (e.g., "过年好", "新年") AND kinship terms (e.g., "姥姥", "奶奶") are detected in audio:
         a) These terms almost certainly refer to people visible in the scene
         b) Search [CANDIDATE MEMORIES] for entities matching these kinship terms
         c) Match the person in visual data to the memory entry with the corresponding `canonical_label`
         d) Use the exact `canonical_label` from memory as the `resolved_identity`
       
       - **Professional Context**: For business settings, require explicit evidence (name mention, introduction, name tag visible)
       
       - **General Rule**: ALWAYS prioritize matching to existing memory entries over creating new identities

    2. **Evidence Triangulation**:
       - Audio transcript provides PRIMARY identity clues (names, kinship terms, greetings)
       - Visual data provides SUPPORTING evidence (appearance, age, gender, clothing)
       - Memory provides HISTORICAL CONTEXT for disambiguation
       - When all three align, confidence should be HIGH

    3. **Context-Aware Inference**:
       - Festive occasions (Spring Festival, etc.) typically involve family gatherings
       - Kinship terms in festive contexts strongly indicate family member presence
       - Match visual person characteristics (age, gender) with kinship term expectations
       - Example: "姥姥" (maternal grandmother) → expect elderly female in visual data

    4. **Logical Chain Documentation**:
       - Cite specific evidence: exact audio phrases, visual characteristics, memory matches
       - Example: "Resolved person_1 as '姥姥' because: (1) Audio contains '过年好' (Spring Festival greeting), (2) Audio mentions kinship context, (3) Visual shows elderly female, (4) Memory entry with canonical_label='姥姥' matches this profile and festive context"

    5. **Memory Action Decision**:
       - Use 'UPDATE' when matching current observations to existing memory (HIGH PRIORITY for kinship matches)
       - Use 'CREATE_NEW' ONLY when absolutely no memory match exists
       - Use 'IGNORE' when evidence is insufficient

    6. **Comprehensive Scene Analysis (WHO, WHAT, WHERE, WHEN)**:
       - WHO: Resolve identities using memory whenever possible
       - WHAT: Determine activity from audio + visual
       - WHERE: Identify location from visual context
       - WHEN: Use festive markers, lighting, and temporal clues

    [OUTPUT SCHEMA (Strict JSON)]
    {{
      "scene_analysis": {{
        "who": [
          {{
            "label_id": "Visual identifier from current frame",
            "resolved_identity": "Canonical identity from memory (USE EXACT canonical_label when matched)",
            "confidence": "high/medium/low",
            "role": "Their role or relationship in this scene",
            "evidence": "Detailed chain: (1) Audio evidence, (2) Visual evidence, (3) Memory match details"
          }}
        ],
        "what": {{
          "activity": "Primary activity or event occurring",
          "interaction_type": "Type of interaction (conversation, greeting, meeting, celebration, etc.)",
          "description": "Detailed description of what is happening"
        }},
        "where": {{
          "location": "Specific location (home, office, hotel, outdoor, etc.)",
          "environment": "Environmental details from visual data",
          "context_clues": "Objects or settings that indicate location"
        }},
        "when": {{
          "time_of_day": "Morning/Afternoon/Evening/Night or specific time if available",
          "season": "Spring/Summer/Fall/Winter",
          "occasion": "Special event, festival, or regular day",
          "temporal_markers": "Audio or visual clues indicating time context"
        }}
      }},
      "evidence_found": {{ 
        "visual": "Key visual observations relevant to identity/context",
        "audio": "Key phrases or mentions from transcript relevant to identity/context"
      }},
      "memory_validation": {{
        "is_relevant": boolean,
        "matched_entities": ["List of canonical_labels matched from memory"],
        "reason": "Explanation of how audio/visual/memory evidence connects"
      }},
      "event_summary": "A coherent narrative summary combining WHO, WHAT, WHERE, WHEN",
      "memory_action": "CREATE_NEW | UPDATE | IGNORE"
    }}
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5  # Lower temperature for more consistent reasoning
    )
    
    raw_content = response.choices[0].message.content
    try:
        json_match = re.search(r'(\{.*\})', raw_content, re.DOTALL)
        return json.loads(json_match.group(1)) if json_match else json.loads(raw_content)
    except Exception as e:
        logger.error(f"JSON Parse Error: {e}")
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_reasoning(video_stem: str):
    logger.info(f"Processing Intelligent Reasoning for: {video_stem}")

    # 1. Load Data
    visual, audio = load_multimodal_data(video_stem)

    # 2. Enhanced Memory Retrieval with Context Analysis
    clues, context_info = extract_dynamic_clues(visual, audio)
    memory_matches = enhanced_memory_retrieval(clues, context_info)

    # 3. Fusion & Inference
    final_analysis = run_fused_reasoning(video_stem, visual, audio, memory_matches, context_info)

    if final_analysis:
        is_relevant = final_analysis.get("memory_validation", {}).get("is_relevant", False)
        action = final_analysis.get("memory_action")
        
        # Store new entities or update existing ones
        if not is_relevant or action == "CREATE_NEW":
            logger.info("New situation detected. Storing inferred data.")
            
            scene = final_analysis.get("scene_analysis", {})
            for person in scene.get("who", []):
                if person.get("resolved_identity"):
                    upsert_memory({
                        "type": "entity",
                        "canonical_label": person["resolved_identity"],
                        "content": person["evidence"],
                        "metadata": {
                            "video": video_stem, 
                            "method": "inference", 
                            "role": person.get("role", ""),
                            "confidence": person.get("confidence", "medium")
                        }
                    })
            
            # Store episodic memory with full context
            what = scene.get("what", {})
            where = scene.get("where", {})
            when = scene.get("when", {})
            
            upsert_memory({
                "type": "episodic",
                "canonical_label": f"Event: {what.get('activity', 'Unknown activity')}",
                "content": final_analysis.get("event_summary", ""),
                "metadata": {
                    "video": video_stem,
                    "location": where.get("location", ""),
                    "occasion": when.get("occasion", ""),
                    "festival": context_info["festive"]["festival"] if context_info["festive"]["festival"] else "",
                    "participants": [p.get("resolved_identity") for p in scene.get("who", [])]
                }
            })
        elif action == "UPDATE":
            logger.info("Updating existing memory entries with new context.")
            # Update existing memories with new observations
            scene = final_analysis.get("scene_analysis", {})
            for person in scene.get("who", []):
                if person.get("resolved_identity") and person.get("confidence") == "high":
                    # Find and update the existing memory
                    existing = find_by_text(person["resolved_identity"], mem_type="entity", top_k=1)
                    if existing:
                        updated_content = f"{existing[0].get('content', '')} | New observation: {person['evidence']}"
                        upsert_memory({
                            "type": "entity",
                            "canonical_label": person["resolved_identity"],
                            "content": updated_content,
                            "metadata": {
                                "video": video_stem,
                                "method": "update",
                                "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        })

        # 4. Save results
        out_file = REASONING_DIR / f"{video_stem}_final_reasoning.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(final_analysis, f, indent=2, ensure_ascii=False)

        # 5. Print formatted analysis
        print("\n" + "="*70)
        print(f"SCENE ANALYSIS FOR: {video_stem}")
        print("="*70)
        
        # Display context information
        if context_info["festive"]["festival"]:
            print(f"\n[CONTEXT] Festival Detected: {context_info['festive']['festival']}")
            print(f"  Markers: {', '.join(context_info['festive']['markers'])}")
        
        if context_info["kinship_terms"]:
            print(f"\n[CONTEXT] Kinship Terms: {', '.join(context_info['kinship_terms'])}")
        
        scene = final_analysis.get("scene_analysis", {})
        
        print("\n[WHO] - People Present:")
        for person in scene.get("who", []):
            confidence = person.get('confidence', 'N/A')
            print(f"  • {person.get('resolved_identity', 'Unknown')} ({person.get('role', 'N/A')}) [Confidence: {confidence}]")
            print(f"    Evidence: {person.get('evidence', 'N/A')}")
        
        print("\n[WHAT] - Activity:")
        what = scene.get("what", {})
        print(f"  • Activity: {what.get('activity', 'N/A')}")
        print(f"  • Type: {what.get('interaction_type', 'N/A')}")
        print(f"  • Description: {what.get('description', 'N/A')}")
        
        print("\n[WHERE] - Location:")
        where = scene.get("where", {})
        print(f"  • Location: {where.get('location', 'N/A')}")
        print(f"  • Environment: {where.get('environment', 'N/A')}")
        print(f"  • Context Clues: {where.get('context_clues', 'N/A')}")
        
        print("\n[WHEN] - Time Context:")
        when = scene.get("when", {})
        print(f"  • Time of Day: {when.get('time_of_day', 'N/A')}")
        print(f"  • Season: {when.get('season', 'N/A')}")
        print(f"  • Occasion: {when.get('occasion', 'N/A')}")
        print(f"  • Temporal Markers: {when.get('temporal_markers', 'N/A')}")
        
        print("\n[SUMMARY]")
        print(f"  {final_analysis.get('event_summary', 'N/A')}")
        
        print("\n[MEMORY VALIDATION]")
        mem_val = final_analysis.get("memory_validation", {})
        print(f"  • Relevant: {mem_val.get('is_relevant', 'N/A')}")
        print(f"  • Matched Entities: {', '.join(mem_val.get('matched_entities', [])) if mem_val.get('matched_entities') else 'None'}")
        print(f"  • Reason: {mem_val.get('reason', 'N/A')}")
        print(f"  • Action: {final_analysis.get('memory_action', 'N/A')}")
        
        print("\n" + "="*70)
        print(f"Full JSON saved to: {out_file}")
        print("="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_reasoning(sys.argv[1])
    else:
        print("Usage: python3 reasoning_agent.py <video_name>")