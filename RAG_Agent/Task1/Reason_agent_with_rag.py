# reasoning_agent_with_rag.py
import os
import json
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Any
import re
import asyncio

# Memory imports
from memory_store import upsert_memory, find_by_text
from clip_embed import get_image_embedding

# RAG-Anything imports
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENAI_API_BASE = "http://127.0.0.1:25000/v1"
MODEL_NAME = "/models/Qwen2.5-VL-7B-Instruct"
API_KEY = "EMPTY"

VISUAL_RESULTS_DIR = "/workspace/test/visual_results"
AUDIO_RESULTS_DIR = "/workspace/test/test_data/transcripts"
OUTPUT_DIR = "/workspace/test/reasoning_results"
VIDEO_DIR = "/workspace/test/test_data"
RAG_WORKING_DIR = "./rag_storage"

FPS = 1
MAX_PROMPT_BATCHES = 40
MAX_TOKENS = 8192

os.makedirs(OUTPUT_DIR, exist_ok=True)

# OpenAI client
client = OpenAI(base_url=OPENAI_API_BASE, api_key=API_KEY)

# Global RAG instance
rag = None

# Global embedding model
_embedding_model = None

# ============================================================================
# EMBEDDING FUNCTIONS
# ============================================================================

def get_embedding_model():
    """Lazy load embedding model"""
    global _embedding_model
    if _embedding_model is None:
        model_path = "/workspace/qwen/agents/RAG_agents/models/paraphrase-multilingual-MiniLM-L12-v2"
        print(f"Loading embedding model from: {model_path}")
        _embedding_model = SentenceTransformer(model_path)
        print("✓ Embedding model loaded")
    return _embedding_model

async def local_embed_func(texts):
    """Local embedding function for RAG"""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()

# ============================================================================
# LLM FUNCTIONS FOR RAG
# ============================================================================

def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """LLM function for RAG text processing"""
    return openai_complete_if_cache(
        MODEL_NAME,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=OPENAI_API_BASE,
        **kwargs,
    )

def vision_model_func(prompt, system_prompt=None, history_messages=[], 
                      image_data=None, messages=None, **kwargs):
    """Vision model function for RAG multimodal processing"""
    if messages:
        return openai_complete_if_cache(
            MODEL_NAME,
            "",
            system_prompt=None,
            history_messages=[],
            messages=messages,
            api_key=API_KEY,
            base_url=OPENAI_API_BASE,
            **kwargs,
        )
    elif image_data:
        content = [{"type": "text", "text": prompt}]
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        })
        
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": content})
        
        return openai_complete_if_cache(
            MODEL_NAME,
            "",
            system_prompt=None,
            history_messages=[],
            messages=msgs,
            api_key=API_KEY,
            base_url=OPENAI_API_BASE,
            **kwargs,
        )
    else:
        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

# ============================================================================
# RAG INITIALIZATION
# ============================================================================

def init_rag():
    """Initialize RAG system"""
    global rag
    
    cfg = RAGAnythingConfig(
        working_dir=RAG_WORKING_DIR,
        parser=None,
        enable_image_processing=True,
        enable_table_processing=False,
        enable_equation_processing=False,
    )
    
    embedding_func = EmbeddingFunc(
        embedding_dim=384,
        max_token_size=512,
        func=local_embed_func,
    )
    
    async def _init():
        global rag
        rag = RAGAnything(
            config=cfg,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func
        )
    
    asyncio.run(_init())

async def async_rag_query(text, mode="hybrid", vlm_enhanced=False, top_k=5):
    """Query RAG knowledge graph"""
    try:
        result = await rag.aquery(text, mode=mode, vlm_enhanced=vlm_enhanced, top_k=top_k)
        return result
    except Exception as e:
        print(f"  Warning: RAG query failed: {e}")
        return None

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def get_video_stem_names(directory):
    """Get all unique video stem names from directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    video_files = []
    p = Path(directory)
    
    for ext in video_extensions:
        video_files.extend(p.glob(f"*{ext}"))
        video_files.extend(p.glob(f"*{ext.upper()}"))
    
    return sorted(set([vf.stem for vf in video_files]))

def extract_range_from_filename(stem: str):
    """Extract frame range from filename"""
    # Pattern: video_frames_000000_000004
    m = re.search(r'_frames[_\-](\d+)[_\-](\d+)$', stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    
    # Pattern: video_frame_0
    m2 = re.search(r'[_\-]frame[s]?[_\-]?(\d+)$', stem)
    if m2:
        v = int(m2.group(1))
        return v, v
    
    return None, None

def load_visual_frames(video_stem, visual_dir):
    """Load all visual analysis JSON results for a video"""
    visual_dir_path = Path(visual_dir)
    all_json = sorted(visual_dir_path.glob("*.json"))
    
    # Filter files matching video_stem
    frame_files = [p for p in all_json if (video_stem in p.stem and "_frames_" in p.stem)]
    
    if not frame_files:
        frame_files = [p for p in all_json if video_stem in p.stem]
    
    if not frame_files:
        frame_files = all_json

    parsed_batches = []
    
    for frame_file in frame_files:
        try:
            with open(frame_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # Remove markdown code blocks
            content = re.sub(r'```(?:json)?\s*', '', content)
            content = re.sub(r'\s*```', '', content)

            # Try direct JSON parse
            frame_data = None
            try:
                frame_data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from text
                match = re.search(r'(\{.*\})', content, re.DOTALL)
                if match:
                    try:
                        frame_data = json.loads(match.group(1))
                    except:
                        frame_data = content

            if frame_data is None:
                frame_data = content

            # Extract frame range from data or filename
            start, end = None, None
            if isinstance(frame_data, dict):
                start = frame_data.get("frame_start")
                end = frame_data.get("frame_end")

            if start is None or end is None:
                fn_start, fn_end = extract_range_from_filename(frame_file.stem)
                if start is None:
                    start = fn_start
                if end is None:
                    end = fn_end

            # Update data dict
            if isinstance(frame_data, dict):
                frame_data.setdefault('video', video_stem)
                if 'frame_start' not in frame_data:
                    frame_data['frame_start'] = start
                if 'frame_end' not in frame_data:
                    frame_data['frame_end'] = end

            parsed_batches.append({
                'video': video_stem,
                'frame_file': frame_file.name,
                'frame_start': start,
                'frame_end': end,
                'data': frame_data
            })

        except Exception as e:
            print(f"Warning: Could not load {frame_file}: {e}")

    # Sort by frame_start
    parsed_batches.sort(key=lambda b: (
        b['frame_start'] if b['frame_start'] is not None else float('inf'),
        b['frame_end'] if b['frame_end'] is not None else float('inf'),
        b['frame_file']
    ))

    return parsed_batches

def load_audio_transcript(video_stem, audio_dir):
    """Load audio transcript JSON for a video"""
    audio_json_path = Path(audio_dir) / f"{video_stem}_transcript.json"
    
    if audio_json_path.exists():
        try:
            with open(audio_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading audio transcript {audio_json_path}: {e}")
            return None
    else:
        print(f"Warning: Audio transcript not found: {audio_json_path}")
        return None

# ============================================================================
# REASONING PROMPT GENERATION
# ============================================================================

def frame_to_time(frame_idx, fps=FPS):
    """Convert frame index to timestamp"""
    try:
        return float(frame_idx) / float(fps)
    except:
        return None

def create_reasoning_prompt(visual_results: List[Dict], audio_data, fps: float = FPS) -> str:
    """Build multimodal reasoning prompt"""
    
    header = (
        "You are an AI agent analyzing first-person AR glasses footage.\n\n"
        "CONTEXT: You observe what the USER SEES (not the user themselves). "
        "The user wears AR glasses capturing their viewpoint.\n\n"
        "TASK: Analyze visual batches + audio to understand:\n"
        "- WHO the user sees\n"
        "- WHAT is happening\n"
        "- WHERE the user is\n"
        "- WHEN events occur\n"
        "- User behaviors and interactions\n\n"
        "Output a comprehensive JSON analysis.\n\n"
    )

    prompt = header
    prompt += "\n--- VISUAL BATCHES ---\n"

    total_batches = len(visual_results)
    if total_batches == 0:
        prompt += "No visual data provided.\n"
    else:
        # Limit batches to avoid token overflow
        if total_batches > MAX_PROMPT_BATCHES:
            # Include first 4, last 4, and sample middle
            selected = list(range(0, min(4, total_batches)))
            selected.extend(range(max(4, total_batches - 4), total_batches))
            
            remaining = MAX_PROMPT_BATCHES - len(selected)
            if remaining > 0 and total_batches > 8:
                step = max(1, (total_batches - 8) // remaining)
                for i in range(4, total_batches - 4, step):
                    if len(selected) >= MAX_PROMPT_BATCHES:
                        break
                    selected.append(i)
            
            included_indices = sorted(set(selected))
            prompt += f"(Total: {total_batches} batches. Including {len(included_indices)} in prompt.)\n\n"
        else:
            included_indices = list(range(total_batches))

        for idx in included_indices:
            vr = visual_results[idx]
            fs = vr.get("frame_start")
            fe = vr.get("frame_end")
            
            tstart = frame_to_time(fs, fps) if fs is not None else None
            tend = frame_to_time(fe, fps) if fe is not None else None
            
            time_str = f"{tstart:.2f}-{tend:.2f}s" if (tstart and tend) else "unknown"
            batch_id = f"Batch {idx+1}: frames={fs}-{fe} time={time_str}"
            
            prompt += f"\n-- {batch_id} --\n"
            try:
                prompt += json.dumps(vr.get("data", vr), ensure_ascii=False, indent=2)
            except:
                prompt += str(vr.get("data", vr))
            prompt += "\n"

    prompt += "\n\n--- AUDIO TRANSCRIPTS ---\n"
    if audio_data:
        try:
            segments = audio_data.get('segments', [])
            # Normalize timestamps
            normalized_segments = []
            for s in segments:
                s_copy = dict(s)
                # Convert ms to seconds if needed
                if 'start_ms' in s_copy:
                    s_copy['start'] = float(s_copy['start_ms']) / 1000.0
                if 'end_ms' in s_copy:
                    s_copy['end'] = float(s_copy['end_ms']) / 1000.0
                normalized_segments.append(s_copy)
            
            prompt += json.dumps({"segments": normalized_segments}, ensure_ascii=False, indent=2)
        except:
            prompt += str(audio_data)
    else:
        prompt += "No audio transcript provided.\n"

    prompt += (
        "\n\n=== INSTRUCTIONS ===\n"
        "1. Analyze ALL audio segments first for context\n"
        "2. Cross-reference with visual batches\n"
        "3. Identify people, objects, locations, activities\n"
        "4. Output valid JSON with your analysis\n"
        "5. Include evidence (batch IDs, timestamps) and confidence scores (0.0-1.0)\n\n"
        "Return ONLY valid JSON. No markdown, no extra text.\n"
    )

    return prompt

# ============================================================================
# JSON EXTRACTION
# ============================================================================

def extract_json_from_text(text: str):
    """Extract first valid JSON object from text"""
    decoder = json.JSONDecoder()
    text = text.strip()
    
    # Try direct parse
    try:
    	   return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object
    idx = 0
    while idx < len(text):
        try:
            obj, end = decoder.raw_decode(text[idx:])
            return obj
        except json.JSONDecodeError:
            next_brace = text.find('{', idx + 1)
            if next_brace == -1:
                break
            idx = next_brace
    
    # Try to fix common issues
    # Remove trailing commas
    cleaned = re.sub(r',\s*(\}|])', r'\1', text)
    try:
        return json.loads(cleaned)
    except:
        pass
    
    return None

# ============================================================================
# REASONING AGENT
# ============================================================================

def run_reasoning_agent(video_stem: str, visual_results: List[Dict], 
                        audio_data, rag_context: str = None, fps: float = FPS) -> Dict[str, Any]:
    """Run the reasoning agent with optional RAG context"""
    print(f"\nRunning Reasoning Agent for: {video_stem}")
    
    prompt = create_reasoning_prompt(visual_results, audio_data, fps=fps)

    # Inject RAG context if available
    if rag_context:
        prompt = (
            "=== KNOWLEDGE GRAPH CONTEXT ===\n"
            f"{rag_context}\n"
            "=== END CONTEXT ===\n\n"
            f"{prompt}"
        )

    # Use simple chat completion (no response_format)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing first-person video footage. Always output valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=MAX_TOKENS
        )

        result_text = response.choices[0].message.content
        
        # Try to extract JSON
        result_json = extract_json_from_text(result_text)
        
        if result_json:
            return result_json
        else:
            return {
                "error": "JSON parsing failed",
                "raw_output": result_text[:1000],
                "video_id": video_stem
            }

    except Exception as e:
        return {
            "error": str(e),
            "video_id": video_stem
        }

# ============================================================================
# MEMORY EXTRACTION
# ============================================================================

def extract_and_persist_memories(video_stem: str, visual_results: List[Dict], 
                                 audio_data, result: Dict[str, Any]):
    """Extract memory candidates from reasoning result and persist to memory store"""
    print(f"\n[MEMORY] Extracting memories from reasoning result for {video_stem}...")
    
    if "error" in result:
        print("[MEMORY] Skipping memory extraction due to reasoning error")
        return
    
    memories_created = 0
    
    try:
        # 1. Extract person names from audio
        all_audio_text = ""
        if audio_data:
            segments = audio_data.get('segments', [])
            all_audio_text = " ".join([seg.get('text', '') for seg in segments])
        
        # Pattern: "叫她/他 [名字]"
        name_pattern = r"(叫她|叫他|你要叫她|你要叫他)\s*([\u4e00-\u9fff]{2,4})"
        matches = re.findall(name_pattern, all_audio_text)
        
        person_names_found = set()
        for match in matches:
            name = match[1].strip()
            person_names_found.add(name)
            print(f"  Found person name in audio: {name}")
        
        # 2. Extract from reasoning result
        who_visible = result.get("1a_external_surroundings", {}).get("who_visible", [])
        if not who_visible:
            # Try alternative keys
            who_visible = result.get("who_visible", [])
        
        for person in who_visible:
            if isinstance(person, dict):
                relationship = person.get("relationship", "")
                name = person.get("name", "")
                
                if name:
                    person_names_found.add(name)
                
                # Family relationships
                if any(term in relationship for term in [
                    "妈妈", "爸爸", "姥姥", "奶奶", "爷爷",
                    "姐姐", "哥哥", "弟弟", "妹妹", "朋友"
                ]):
                    person_names_found.add(relationship)
        
        # 3. Create person_alias memories
        for name in person_names_found:
            existing = find_by_text(name, mem_type="person_alias")
            
            evidence_items = [f"mentioned in {video_stem}"]
            if who_visible:
                evidence_items.append(f"appeared in {len(who_visible)} scenes")
            
            mem = {
                "type": "person_alias",
                "canonical_label": name,
                "aliases": [],
                "metadata": {
                    "videos": [video_stem],
                    "first_seen": video_stem,
                    "relationship_context": "family/friend"
                },
                "evidence": evidence_items,
                "confidence": 0.85 if len(evidence_items) > 1 else 0.6,
                "image_vec": None
            }
            
            mem_id = upsert_memory(mem)
            memories_created += 1
            print(f"  ✓ Created/updated person memory: {name} (ID: {mem_id})")
        
        # 4. Extract event memories
        event_keywords = ["拜年", "生日", "聚会", "旅游", "会议", "上课", "吃饭", "扫雪", "打扫"]
        events_found = []
        
        for keyword in event_keywords:
            if keyword in all_audio_text:
                events_found.append(keyword)
        
        # Extract from reasoning
        what_happening = result.get("1a_external_surroundings", {}).get("what_happening", {})
        if not what_happening:
            what_happening = result.get("what_happening", {})
        
        if isinstance(what_happening, dict):
            events_from_reasoning = what_happening.get("events", [])
            for event_desc in events_from_reasoning:
                if isinstance(event_desc, str):
                    events_found.append(event_desc)
        
        for event in set(events_found):
            mem = {
                "type": "event",
                "canonical_label": event,
                "aliases": [],
                "metadata": {
                    "video": video_stem,
                    "context": "detected_from_reasoning"
                },
                "evidence": [f"observed in {video_stem}"],
                "confidence": 0.75,
                "image_vec": None
            }
            
            mem_id = upsert_memory(mem)
            memories_created += 1
            print(f"  ✓ Created event memory: {event} (ID: {mem_id})")
        
        # 5. Extract place memories
        where_info = result.get("1a_external_surroundings", {}).get("where", {})
        if not where_info:
            where_info = result.get("where", {})
        
        if isinstance(where_info, dict):
            place_type = where_info.get("place_type", "")
            place_name = where_info.get("place_name", "")
            
            if place_name and place_name.lower() not in ["null", "unknown", ""]:
                mem = {
                    "type": "place",
                    "canonical_label": place_name,
                    "aliases": [place_type] if place_type else [],
                    "metadata": {
                        "video": video_stem,
                        "place_type": place_type,
                        "details": where_info.get("details", "")
                    },
                    "evidence": [f"identified in {video_stem}"],
                    "confidence": where_info.get("confidence", 0.7),
                    "image_vec": None
                }
                
                mem_id = upsert_memory(mem)
                memories_created += 1
                print(f"  ✓ Created place memory: {place_name} (ID: {mem_id})")
        
        print(f"[MEMORY] Total memories created/updated: {memories_created}")
        
    except Exception as e:
        print(f"[MEMORY] Error during memory extraction: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_reasoning_result(video_stem: str, result: Dict[str, Any], output_dir: str):
    """Save reasoning result to JSON file"""
    output_path = Path(output_dir) / f"{video_stem}_reasoning.json"

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved reasoning result to: {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Error saving result: {e}")
        return None

def generate_summary_report(all_results: List[Dict], output_dir: str):
    """Generate a summary report of all reasoning results"""
    summary_path = Path(output_dir) / "reasoning_summary.json"

    summary = {
        "total_videos": len(all_results),
        "successful": sum(1 for r in all_results if "error" not in r.get("result", {})),
        "failed": sum(1 for r in all_results if "error" in r.get("result", {})),
        "videos": []
    }

    for res in all_results:
        video_summary = {
            "video_name": res["video_stem"],
            "status": "success" if "error" not in res.get("result", {}) else "failed",
            "visual_batches": res.get("visual_frames_count", 0),
            "audio_segments": res.get("audio_segments_count", 0)
        }

        if "error" not in res.get("result", {}):
            result = res["result"]
            # Extract key fields (flexible)
            if "summary" in result:
                video_summary["summary"] = result["summary"]
            if "conclusion" in result:
                video_summary["conclusion"] = result["conclusion"]
            if "key_events" in result:
                video_summary["key_events"] = result["key_events"]
        else:
            video_summary["error"] = res["result"].get("error", "Unknown error")

        summary["videos"].append(video_summary)

    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Summary report saved to: {summary_path}")
    except Exception as e:
        print(f"\n✗ Error saving summary: {e}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*70)
    print("MULTI-MODAL REASONING AGENT (with RAG)")
    print("="*70)
    print(f"Visual results dir: {VISUAL_RESULTS_DIR}")
    print(f"Audio results dir:  {AUDIO_RESULTS_DIR}")
    print(f"Output dir:         {OUTPUT_DIR}")
    print("="*70)

    # Initialize RAG
    try:
        print("\nInitializing RAG...")
        init_rag()
        print("✓ RAG initialized")
    except Exception as e:
        print(f"Warning: Failed to init RAG: {e}. Continuing without RAG.")
    
    # Determine video stems
    video_stems = get_video_stem_names(VIDEO_DIR)
    
    if not video_stems:
        # Fallback: extract from visual result filenames
        p = Path(VISUAL_RESULTS_DIR)
        stems = set()
        for fn in sorted(p.glob("*.json")):
            m = re.match(r'(?P<video>.+?)_frames[_\-]\d+[_\-]\d+$', fn.stem)
            if m:
                stems.add(m.group('video'))
            else:
                m2 = re.match(r'(?P<video>.+?)_frame[s]?[_\-]?\d+$', fn.stem)
                if m2:
                    stems.add(m2.group('video'))
        video_stems = sorted(stems)

    if not video_stems:
        print("✗ No videos found")
        return

    print(f"\nFound {len(video_stems)} video(s) to process\n")

    all_results = []
    success_count = 0
    failed_count = 0

    for idx, video_stem in enumerate(video_stems, 1):
        print(f"\n{'='*70}")
        print(f"Processing [{idx}/{len(video_stems)}]: {video_stem}")
        print(f"{'='*70}")

        try:
            # Load visual results
            print("  [1/5] Loading visual analysis results...")
            visual_results = load_visual_frames(video_stem, VISUAL_RESULTS_DIR)
            print(f"  ✓ Loaded {len(visual_results)} visual batch(es)")

            # Load audio transcript
            print("  [2/5] Loading audio transcript...")
            audio_data = load_audio_transcript(video_stem, AUDIO_RESULTS_DIR)
            audio_segments = len(audio_data.get('segments', [])) if audio_data else 0
            print(f"  ✓ Loaded {audio_segments} audio segment(s)")

            # Query RAG for context
            rag_context = None
            if rag is not None:
                try:
                    print("  [3/5] Querying RAG for context...")
                    query_text = (
                        f"Retrieve context for video '{video_stem}': "
                        f"timeline, key frames, transcripts, entities, relationships"
                    )
                    rag_resp = asyncio.run(async_rag_query(
                        query_text, 
                        mode="hybrid", 
                        vlm_enhanced=False, 
                        top_k=8
                    ))
                    
                    if rag_resp:
                        if isinstance(rag_resp, (dict, list)):
                            rag_context = json.dumps(rag_resp, ensure_ascii=False)[:10000]
                        else:
                            rag_context = str(rag_resp)[:10000]
                        print("  ✓ Retrieved RAG context")
                    else:
                        print("  ✗ RAG query returned no results")
                except Exception as e:
                    print(f"  ✗ RAG query failed: {e}")
            else:
                print("  [3/5] Skipping RAG query (RAG not initialized)")

            # Run reasoning
            print("  [4/5] Running reasoning agent...")
            result = run_reasoning_agent(
                video_stem, 
                visual_results, 
                audio_data, 
                rag_context=rag_context, 
                fps=FPS
            )

            if "error" in result:
                print(f"  ✗ Reasoning failed: {result.get('error', 'Unknown error')}")
                failed_count += 1
            else:
                print("  ✓ Reasoning complete")
                success_count += 1

            # Extract and persist memories
            print("  [5/5] Extracting and persisting memories...")
            extract_and_persist_memories(video_stem, visual_results, audio_data, result)

            # Save result
            save_reasoning_result(video_stem, result, OUTPUT_DIR)

            all_results.append({
                "video_stem": video_stem,
                "result": result,
                "visual_frames_count": len(visual_results),
                "audio_segments_count": audio_segments
            })

            if "error" not in result:
                print(f"✓ SUCCESS: {video_stem}")
            else:
                print(f"✗ FAILED: {video_stem}")

        except Exception as e:
            print(f"✗ Error processing {video_stem}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            all_results.append({
                "video_stem": video_stem,
                "result": {"error": str(e)},
                "visual_frames_count": 0,
                "audio_segments_count": 0
            })

    # Generate summary report
    print("\n" + "="*70)
    print("Generating summary report...")
    generate_summary_report(all_results, OUTPUT_DIR)

    print(f"\n{'='*70}")
    print("REASONING AGENT PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successful: {success_count}/{len(video_stems)}")
    print(f"✗ Failed:     {failed_count}/{len(video_stems)}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()