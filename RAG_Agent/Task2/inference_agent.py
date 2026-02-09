'''Task 2 Reasoning'''
import os
os.environ["HF_ENDPOINT"] = "https://www.modelscope.cn"
os.environ["TRANSFORMERS_NO_TF"] = "1"
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pickle
from pathlib import Path


class MemoryDatabase:
    def __init__(self, embedding_dim: int = 384, model_name: str = "all-MiniLM-L6-v2",
                 persist_dir: str = "./memory_store"):
        self.embedding_dim = embedding_dim
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

        self.memories: Dict[str, Dict[str, Any]] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata_index: Dict[str, List[str]] = {
            "episodic": [], "semantic": [], "preference": []
        }

        # Load embedding model
        self.embed_model = SentenceTransformer("./all-MiniLM-L6-v2", device="cpu")

        # Load existing memories if available
        self.load_from_disk()

    def save_to_disk(self):
        """Persist all memories to disk."""
        try:
            # Save memories as JSON
            memories_path = self.persist_dir / "memories.json"
            with open(memories_path, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, indent=2, ensure_ascii=False)

            # Save vectors as pickle (numpy arrays)
            vectors_path = self.persist_dir / "vectors.pkl"
            with open(vectors_path, 'wb') as f:
                pickle.dump(self.vectors, f)

            # Save metadata index
            index_path = self.persist_dir / "metadata_index.json"
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_index, f, indent=2)

            print(f"✓ Saved {len(self.memories)} memories to {self.persist_dir}")
            return True
        except Exception as e:
            print(f"✗ Error saving memories: {e}")
            return False

    def load_from_disk(self):
        """Load memories from disk if they exist."""
        try:
            memories_path = self.persist_dir / "memories.json"
            vectors_path = self.persist_dir / "vectors.pkl"
            index_path = self.persist_dir / "metadata_index.json"

            if not memories_path.exists():
                print(f"No existing memories found in {self.persist_dir}")
                return False

            # Load memories
            with open(memories_path, 'r', encoding='utf-8') as f:
                self.memories = json.load(f)

            # Load vectors
            with open(vectors_path, 'rb') as f:
                self.vectors = pickle.load(f)

            # Load metadata index
            with open(index_path, 'r', encoding='utf-8') as f:
                self.metadata_index = json.load(f)

            print(f"✓ Loaded {len(self.memories)} memories from {self.persist_dir}")
            return True
        except Exception as e:
            print(f"✗ Error loading memories: {e}")
            return False

    def upsert_memory(self, memory: Dict[str, Any], vector: Optional[np.ndarray] = None,
                      auto_save: bool = True) -> str:
        """Insert or update a memory record."""
        mem_id = memory.get("id")
        if not mem_id:
            mem_id = str(uuid.uuid4())
            memory["id"] = mem_id

        if vector is None:
            canonical = self._canonicalize_memory(memory)
            vector = self._embed(canonical)

        self.memories[mem_id] = memory
        self.vectors[mem_id] = vector

        mem_type = memory.get("type", "episodic")
        if mem_id not in self.metadata_index.get(mem_type, []):
            self.metadata_index.setdefault(mem_type, []).append(mem_id)

        # Auto-save to disk after each update
        if auto_save:
            self.save_to_disk()

        return mem_id

    def export_json(self, filepath: str):
        """Export all memories to a single JSON file (for backup/sharing)."""
        export_data = {
            "metadata": {
                "total_memories": len(self.memories),
                "export_timestamp": datetime.now().isoformat(),
                "embedding_dim": self.embedding_dim
            },
            "memories": self.memories,
            "metadata_index": self.metadata_index
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Exported {len(self.memories)} memories to {filepath}")

    def import_json(self, filepath: str):
        """Import memories from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        imported_memories = data.get("memories", {})

        for mem_id, memory in imported_memories.items():
            # Re-compute vectors on import
            self.upsert_memory(memory, auto_save=False)

        # Save once after all imports
        self.save_to_disk()
        print(f"✓ Imported {len(imported_memories)} memories from {filepath}")

    """In-memory Vector + Metadata Memory Database for VR Reasoning Agent."""


    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    def _canonicalize_memory(self, memory: Dict[str, Any]) -> str:
        """Build canonical text representation for embedding."""
        intent = memory.get("intent_inference", {})
        what = intent.get("what", {}).get("label", "")
        when = intent.get("when", {}).get("label", "")
        where = intent.get("where", {}).get("label", "")
        how = intent.get("how", {}).get("label", "")

        context_text = memory.get("modalities", {}).get("text", "")
        concepts = ",".join(memory.get("derived_concepts", []))

        canonical = f"what:{what}|when:{when}|where:{where}|how:{how}|context:{context_text}|concepts:{concepts}"
        return canonical

    def _embed(self, text: str) -> np.ndarray:
        """Real embedding function using sentence-transformers."""
        vec = self.embed_model.encode(text, convert_to_numpy=True)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    

    def retrieve_similar(
            self,
            query_snapshot: Dict[str, Any],
            top_k: int = 5,
            type_filter: Optional[List[str]] = None,
            sensitivity_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K similar memories given a query snapshot."""
        query_canonical = self._canonicalize_memory(query_snapshot)
        query_vec = self._embed(query_canonical)

        candidates = []
        for mem_id, mem_vec in self.vectors.items():
            memory = self.memories[mem_id]

            if type_filter and memory.get("type") not in type_filter:
                continue
            if sensitivity_filter:
                sens = memory.get("privacy", {}).get("sensitivity", "low")
                if sens not in sensitivity_filter:
                    continue

            sim = self._cosine_similarity(query_vec, mem_vec)

            mem_time = datetime.fromisoformat(memory["timestamp"].replace("Z", "+00:00"))
            now = datetime.now().astimezone()
            hours_diff = (now - mem_time).total_seconds() / 3600
            recency_decay = np.exp(-0.01 * hours_diff)

            type_bonus = 0.1 if memory.get("type") == "preference" else 0.0
            confidence = memory.get("provenance", {}).get("confidence_overall", 0.5)

            score = 0.6 * sim + 0.2 * recency_decay + 0.1 * type_bonus + 0.1 * confidence

            candidates.append({
                "memory": memory,
                "score": score,
                "similarity": sim,
                "recency_decay": recency_decay
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def get_memory(self, mem_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory by ID."""
        return self.memories.get(mem_id)

    def list_memories(self, mem_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all memories, optionally filtered by type."""
        if mem_type:
            ids = self.metadata_index.get(mem_type, [])
            return [self.memories[mid] for mid in ids if mid in self.memories]
        return list(self.memories.values())


# ---- Demand Recognition Agent ----
class DemandRecognitionAgent:
    """
    LLM-based agent that:
    1. Takes current user situation description
    2. Retrieves similar past memories
    3. Infers user demand (what/when/where/how)
    4. Generates multiple solution options with confidence scores
    """

    def __init__(self, memory_db: MemoryDatabase, openai_client: OpenAI):
        self.memory_db = memory_db
        self.client = openai_client

    def analyze_situation(self, situation_description: str) -> Dict[str, Any]:
        """
        Main entry point: analyze current situation and generate demand inference + solutions.

        Args:
            situation_description: Natural language description of current user situation

        Returns:
            Dictionary containing:
            - current_situation: parsed snapshot
            - retrieved_memories: similar past memories
            - demand_inference: what/when/where/how with confidence
            - solution_options: ranked list of solutions
        """
        print(f"\n{'=' * 70}")
        print("DEMAND RECOGNITION AGENT")
        print(f"{'=' * 70}")
        print(f"Situation: {situation_description[:100]}...")

        # Step 1: Parse situation into structured snapshot
        print("\n[1/4] Parsing situation into structured snapshot...")
        current_snapshot = self._parse_situation(situation_description)

        # Step 2: Retrieve similar memories
        print("\n[2/4] Retrieving similar memories...")
        similar_memories = self.memory_db.retrieve_similar(current_snapshot, top_k=5)
        print(f"  Found {len(similar_memories)} similar memories")

        # Step 3: Infer demand using LLM
        print("\n[3/4] Inferring user demand with LLM...")
        demand_inference = self._infer_demand(current_snapshot, similar_memories)

        # Step 4: Generate solution options
        print("\n[4/4] Generating solution options...")
        solution_options = self._generate_solutions(current_snapshot, demand_inference, similar_memories)

        result = {
            "timestamp": datetime.now().isoformat(),
            "current_situation": current_snapshot,
            "retrieved_memories": [
                {
                    "memory_id": m["memory"]["id"],
                    "score": m["score"],
                    "similarity": m["similarity"],
                    "summary": m["memory"]["modalities"].get("text", "")[:100]
                }
                for m in similar_memories
            ],
            "demand_inference": demand_inference,
            "solution_options": solution_options
        }

        # Store this interaction as new episodic memory
        self._store_interaction_memory(current_snapshot, demand_inference, solution_options)

        return result

    def _parse_situation(self, description: str) -> Dict[str, Any]:
        """Parse natural language situation into structured snapshot."""
        prompt = f"""Parse this AR glasses user situation into structured JSON.

Situation: {description}

Output JSON with this schema:
{{
  "type": "episodic",
  "timestamp": "<ISO timestamp>",
  "source": "user_description",
  "context": {{
    "scene_label": "<location/scene>",
    "relative_location": "<user position>"
  }},
  "modalities": {{
    "text": "<full situation description>",
    "vision": "<visual cues>",
    "gesture": "<any gestures mentioned>"
  }},
  "intent_inference": {{}},
  "derived_concepts": ["<key concepts>"]
}}

Output only valid JSON, no markdown."""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="/models/Qwen2.5-VL-7B-Instruct",
                temperature=0.1,
                max_tokens=1024
            )

            result = response.choices[0].message.content
            
            # Debug output
            print(f"\n🔍 DEBUG - Raw LLM Response:\n{result}\n")
            
            # Clean markdown if present
            result = result.replace("```json", "").replace("```", "").strip()
            snapshot = json.loads(result)
            
            # Ensure required fields exist with defaults
            snapshot.setdefault("type", "episodic")
            snapshot.setdefault("timestamp", datetime.now().isoformat() + "Z")
            snapshot.setdefault("source", "user_description")
            snapshot.setdefault("context", {"scene_label": "unknown"})
            snapshot.setdefault("modalities", {"text": description})
            snapshot.setdefault("intent_inference", {})
            snapshot.setdefault("derived_concepts", [])
            
            # Override timestamp to ensure it's current
            snapshot["timestamp"] = datetime.now().isoformat() + "Z"
            
            return snapshot

        except Exception as e:
            print(f"Warning: LLM parsing failed ({e}), using fallback")
            return {
                "type": "episodic",
                "timestamp": datetime.now().isoformat() + "Z",
                "source": "user_description",
                "context": {"scene_label": "unknown"},
                "modalities": {"text": description},
                "intent_inference": {},
                "derived_concepts": []
            }

    def _infer_demand(self, current_snapshot: Dict, similar_memories: List[Dict]) -> Dict[str, Any]:
        """Use LLM to infer user demand (what/when/where/how) based on current situation + memories."""

        # Build context from memories
        memory_context = ""
        for i, mem in enumerate(similar_memories[:3], 1):
            m = mem["memory"]
            memory_context += f"\nMemory {i} (similarity: {mem['similarity']:.2f}):\n"
            memory_context += f"  Text: {m['modalities'].get('text', '')[:150]}\n"
            intent = m.get("intent_inference", {})
            if intent:
                memory_context += f"  Past intent: what={intent.get('what', {}).get('label', '?')}, "
                memory_context += f"where={intent.get('where', {}).get('label', '?')}, "
                memory_context += f"how={intent.get('how', {}).get('label', '?')}\n"

        prompt = f"""You are an AR glasses AI agent. Analyze the user's current situation and infer their demand.

CURRENT SITUATION:
{json.dumps(current_snapshot, indent=2)}

SIMILAR PAST SITUATIONS:
{memory_context}

Based on the current situation and similar past memories, infer the user's demand:

2.a WHAT user wants (information, content, actions, services)
2.b WHEN user wants it (timing, conditions, triggers)
2.c WHERE user wants it (location, environment, display target)
2.d HOW user wants it (UI/UX, interaction method, notification style)

Output JSON:
{{
  "what": {{
    "label": "<action/service>",
    "description": "<detailed need>",
    "confidence": <0.0-1.0>,
    "evidence": ["<reasons from situation/memories>"]
  }},
  "when": {{
    "label": "<timing>",
    "description": "<when to execute>",
    "confidence": <0.0-1.0>,
    "evidence": ["<reasons>"]
  }},
  "where": {{
    "label": "<location/target>",
    "description": "<where to show/execute>",
    "confidence": <0.0-1.0>,
    "evidence": ["<reasons>"]
  }},
  "how": {{
    "label": "<interaction method>",
    "description": "<UI/UX preference>",
    "confidence": <0.0-1.0>,
    "evidence": ["<reasons>"]
  }}
}}

Output only valid JSON."""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="/models/Qwen2.5-VL-7B-Instruct",
                temperature=0.2,
                max_tokens=2048
            )

            result = response.choices[0].message.content
            result = result.replace("```json", "").replace("```", "").strip()
            return json.loads(result)

        except Exception as e:
            print(f"Warning: Demand inference failed ({e})")
            return {
                "what": {"label": "unknown", "confidence": 0.3, "evidence": ["parsing_error"]},
                "when": {"label": "immediate", "confidence": 0.5, "evidence": []},
                "where": {"label": "current_location", "confidence": 0.5, "evidence": []},
                "how": {"label": "default", "confidence": 0.5, "evidence": []}
            }

    def _generate_solutions(self, current_snapshot: Dict, demand_inference: Dict,
                            similar_memories: List[Dict]) -> List[Dict[str, Any]]:
        """Generate multiple solution options ranked by confidence."""

        # Build memory solutions context
        past_solutions = ""
        for i, mem in enumerate(similar_memories[:3], 1):
            m = mem["memory"]
            actions = m.get("actions_taken", [])
            if actions:
                past_solutions += f"\nMemory {i} solution: {actions[0].get('payload', 'N/A')}\n"
                feedback = m.get("user_feedback", {})
                if feedback:
                    past_solutions += f"  User response: {feedback.get('response', 'N/A')}\n"

        prompt = f"""You are an AR glasses AI agent. Generate solution options for the user's inferred demand.

INFERRED DEMAND:
{json.dumps(demand_inference, indent=2)}

CURRENT SITUATION:
{json.dumps(current_snapshot['modalities'], indent=2)}

PAST SIMILAR SOLUTIONS:
{past_solutions}

Generate 3 solution options ranked by confidence. Each solution should specify:
- Concrete actions to take
- UI/UX implementation details
- Expected outcome
- Confidence score based on evidence strength

Output JSON array:
[
  {{
    "rank": 1,
    "solution_id": "<uuid>",
    "description": "<what the system will do>",
    "actions": [
      {{
        "type": "<action_type>",
        "target": "<where/what>",
        "payload": "<action details>",
        "timing": "<when to execute>"
      }}
    ],
    "ui_ux": {{
      "display_method": "<how to show>",
      "interaction_required": "<user input needed>",
      "notification_style": "<alert/silent/etc>"
    }},
    "expected_outcome": "<what user will achieve>",
    "confidence": <0.0-1.0>,
    "reasoning": "<why this solution fits>",
    "evidence_from_memories": ["<memory references>"]
  }}
]

Output only valid JSON array."""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="/models/Qwen2.5-VL-7B-Instruct",
                temperature=0.3,
                max_tokens=3072
            )

            result = response.choices[0].message.content
            result = result.replace("```json", "").replace("```", "").strip()
            solutions = json.loads(result)

            # Ensure solutions are sorted by confidence
            solutions.sort(key=lambda x: x.get("confidence", 0), reverse=True)

            # Add UUIDs if missing
            for i, sol in enumerate(solutions, 1):
                if "solution_id" not in sol:
                    sol["solution_id"] = str(uuid.uuid4())
                if "rank" not in sol:
                    sol["rank"] = i

            return solutions

        except Exception as e:
            print(f"Warning: Solution generation failed ({e})")
            return [{
                "rank": 1,
                "solution_id": str(uuid.uuid4()),
                "description": "Unable to generate solutions due to parsing error",
                "actions": [],
                "confidence": 0.3,
                "reasoning": f"Error: {str(e)}"
            }]

    def _store_interaction_memory(self, snapshot: Dict, demand: Dict, solutions: List[Dict]):
        """Store this interaction as a new episodic memory for future retrieval."""
        memory = {
            "type": "episodic",
            "timestamp": datetime.now().isoformat() + "Z",
            "source": "demand_recognition_agent",
            "context": snapshot.get("context", {}),
            "modalities": snapshot.get("modalities", {}),
            "intent_inference": {
                "what": demand.get("what", {}),
                "when": demand.get("when", {}),
                "where": demand.get("where", {}),
                "how": demand.get("how", {})
            },
            "actions_taken": [
                {
                    "type": "solution_generated",
                    "payload": sol.get("description", ""),
                    "confidence": sol.get("confidence", 0)
                }
                for sol in solutions[:1]  # Store top solution
            ],
            "derived_concepts": snapshot.get("derived_concepts", []),
            "provenance": {
                "agent": "demand_recognition_agent",
                "confidence_overall": solutions[0].get("confidence", 0.5) if solutions else 0.5
            },
            "privacy": {
                "sensitivity": "medium",
                "retention_policy": "standard"
            }
        }

        mem_id = self.memory_db.upsert_memory(memory)
        print(f"  ✓ Stored interaction as memory: {mem_id}")


# ---- Demo Integration ----
def demo_demand_recognition():
    """Demo: User describes situation, agent retrieves memories and generates solutions."""

    print("\n" + "=" * 70)
    print("INITIALIZING DEMAND RECOGNITION SYSTEM")
    print("=" * 70)

    # Initialize memory database with persistent storage
    print("\n[1/3] Loading memory database...")
    memory_db = MemoryDatabase(
        embedding_dim=384,
        model_name="all-MiniLM-L6-v2",
        persist_dir="./memory_store"  # Persistent storage directory
    )

    # Seed with example memories only if database is empty
    if len(memory_db.memories) == 0:
        print("\n[2/3] Seeding example memories...")
        seed_memories = [
            {
                "type": "preference",
                "timestamp": "2024-01-15T10:30:00Z",
                "source": "user_settings",
                "context": {"scene_label": "classroom"},
                "modalities": {
                    "text": "User prefers slides with speaker script visible and voice readout enabled."
                },
                "intent_inference": {
                    "what": {"label": "present_slides", "confidence": 0.9},
                    "where": {"label": "classroom_projector", "confidence": 0.85},
                    "how": {"label": "voice_and_script", "confidence": 0.95}
                },
                "derived_concepts": ["pref_show_script", "pref_voice_readout"],
                "provenance": {"agent": "preference_learner", "confidence_overall": 0.92},
                "privacy": {"sensitivity": "low", "retention_policy": "long_term"}
            },
            {
                "type": "episodic",
                "timestamp": datetime.now().isoformat() + "Z",
                "source": "multimodal_perception",
                "context": {"scene_label": "classroom_front", "relative_location": "standing_at_podium"},
                "modalities": {
                    "text": "ASR: 'Please present now.' Opened file: climate_presentation.pptx, slide_index:3",
                    "vision": "Teacher standing, students seated, projector screen visible",
                    "gesture": "user_hand_gesture_forward"
                },
                "intent_inference": {
                    "what": {"label": "show_slide", "confidence": 0.88, "evidence": ["ASR_command", "file_opened"]},
                    "when": {"label": "immediate", "confidence": 0.95},
                    "where": {"label": "classroom_projector", "confidence": 0.90},
                    "how": {"label": "mirror_slide_and_show_script", "confidence": 0.85}
                },
                "derived_concepts": ["presentation_mode", "classroom_context"],
                "provenance": {"agent": "multimodal_reasoner", "confidence_overall": 0.89},
                "privacy": {"sensitivity": "medium", "retention_policy": "standard"}
            },
            {
                "type": "episodic",
                "timestamp": "2024-01-10T14:20:00Z",
                "source": "multimodal_perception",
                "context": {"scene_label": "classroom", "relative_location": "student_desk"},
                "modalities": {
                    "text": "User presented slide 2 with script; confirmed action via one-tap; projector mirrored display",
                    "vision": "Slide visible on projector, script on AR display",
                    "gesture": "single_tap_confirm"
                },
                "intent_inference": {
                    "what": {"label": "present_with_notes", "confidence": 0.92},
                    "when": {"label": "immediate", "confidence": 0.95},
                    "where": {"label": "projector_and_ar_overlay", "confidence": 0.88},
                    "how": {"label": "one_tap_gesture", "confidence": 0.90}
                },
                "actions_taken": [
                    {"type": "mirror_display", "target": "projector", "payload": "slide_2_with_script"}
                ],
                "user_feedback": {"response": "positive", "implicit_signal": "continued_presentation"},
                "derived_concepts": ["successful_presentation", "gesture_preference"],
                "provenance": {"agent": "action_executor", "confidence_overall": 0.91},
                "privacy": {"sensitivity": "low", "retention_policy": "standard"}
            },
            {
                "type": "semantic",
                "timestamp": "2024-01-05T09:00:00Z",
                "source": "pattern_learner",
                "context": {"scene_label": "classroom"},
                "modalities": {
                    "text": "User habit: during classroom presentations, user prefers projector mirroring and script visible on AR overlay"
                },
                "intent_inference": {
                    "what": {"label": "presentation_habit", "confidence": 0.85},
                    "where": {"label": "classroom", "confidence": 0.90},
                    "how": {"label": "dual_display", "confidence": 0.88}
                },
                "derived_concepts": ["presentation_pattern", "dual_display_preference"],
                "provenance": {"agent": "pattern_learner", "confidence_overall": 0.87},
                "privacy": {"sensitivity": "low", "retention_policy": "long_term"}
            },
            {
                "type": "episodic",
                "timestamp": "2024-01-08T16:45:00Z",
                "source": "multimodal_perception",
                "context": {"scene_label": "library", "relative_location": "reading_desk"},
                "modalities": {
                    "text": "User reading research paper; highlighted text; wants to save note.",
                    "vision": "Document on table, hand holding pen, highlighting text",
                    "gesture": "highlight_and_hold"
                },
                "intent_inference": {
                    "what": {"label": "save_note", "confidence": 0.90},
                    "when": {"label": "immediate", "confidence": 0.92},
                    "where": {"label": "notes_app", "confidence": 0.88},
                    "how": {"label": "gesture_or_voice", "confidence": 0.85}
                },
                "derived_concepts": ["note_taking", "research_mode"],
                "provenance": {"agent": "multimodal_reasoner", "confidence_overall": 0.89},
                "privacy": {"sensitivity": "medium", "retention_policy": "standard"}
            }
        ]

        # Insert all seed memories without auto-save
        for mem in seed_memories:
            memory_db.upsert_memory(mem, auto_save=False)

        # Save once after all seeds
        memory_db.save_to_disk()
        print(f"  ✓ Seeded {len(seed_memories)} memories")
    else:
        print(f"\n[2/3] Using existing {len(memory_db.memories)} memories from disk")
        print(f"  Episodic: {len(memory_db.metadata_index.get('episodic', []))}")
        print(f"  Semantic: {len(memory_db.metadata_index.get('semantic', []))}")
        print(f"  Preference: {len(memory_db.metadata_index.get('preference', []))}")

    # Initialize OpenAI client
    print("\n[3/3] Connecting to LLM...")
    openai_api_base = "http://127.0.0.1:25000/v1"
    client = OpenAI(base_url=openai_api_base, api_key="EMPTY")

    # Create agent
    agent = DemandRecognitionAgent(memory_db, client)

    print("\n" + "=" * 70)
    print("SYSTEM READY")
    print("=" * 70)

    # Demo scenarios
    scenarios = [
        {
            "name": "Classroom Presentation",
            "description": """Teacher asks user to present climate slide now. User is standing at podium 
            in classroom. User opened PowerPoint file 'climate_presentation.pptx' and is viewing slide 3. 
            Students are seated and waiting. Projector is available."""
        },
        {
            "name": "Library Note Taking",
            "description": """User is in library reading a research paper about AI. User highlighted an 
            important paragraph about neural networks and wants to save it for later review. User's hand 
            is hovering over the highlighted text."""
        },
        {
            "name": "Coffee Shop Meeting",
            "description": """User is sitting in coffee shop with a colleague. Colleague asks 'Can you 
            show me that product demo?' User has laptop open with demo video file. User wants to share 
            screen to colleague's device."""
        }
    ]

    # Run demos
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n\n{'#' * 70}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'#' * 70}")
        print(f"\nSituation: {scenario['description']}")

        result = agent.analyze_situation(scenario['description'])

        # Print results
        print(f"\n{'=' * 70}")
        print("ANALYSIS RESULTS")
        print(f"{'=' * 70}")

        print("\n--- Retrieved Memories ---")
        for mem in result['retrieved_memories']:
            print(f"  • [{mem['memory_id'][:8]}...] (score: {mem['score']:.3f}, sim: {mem['similarity']:.3f})")
            print(f"    {mem['summary'][:80]}...")

        print("\n--- Demand Inference ---")
        demand = result['demand_inference']
        for aspect in ['what', 'when', 'where', 'how']:
            if aspect in demand:
                info = demand[aspect]
                print(f"\n  {aspect.upper()}: {info.get('label', 'N/A')} (confidence: {info.get('confidence', 0):.2f})")
                print(f"    Description: {info.get('description', 'N/A')}")
                evidence = info.get('evidence', [])
                if evidence:
                    print(f"    Evidence: {', '.join(evidence[:2])}")

        print("\n--- Solution Options ---")
        for sol in result['solution_options']:
            print(f"\n  [{sol['rank']}] {sol['description']}")
            print(f"      Confidence: {sol.get('confidence', 0):.2f}")
            print(f"      Reasoning: {sol.get('reasoning', 'N/A')[:100]}...")

            if 'actions' in sol and sol['actions']:
                print(f"      Actions:")
                for action in sol['actions'][:2]:
                    print(f"        - {action.get('type', 'N/A')}: {action.get('payload', 'N/A')[:60]}...")

            if 'ui_ux' in sol:
                ux = sol['ui_ux']
                print(f"      UI/UX: {ux.get('display_method', 'N/A')}, {ux.get('notification_style', 'N/A')}")

        # Save result to file
        output_path = f"demand_recognition_scenario_{i}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Saved detailed result to: {output_path}")

    # Final statistics
    print(f"\n\n{'=' * 70}")
    print("SESSION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total memories in database: {len(memory_db.memories)}")
    print(f"  Episodic: {len(memory_db.metadata_index['episodic'])}")
    print(f"  Semantic: {len(memory_db.metadata_index['semantic'])}")
    print(f"  Preference: {len(memory_db.metadata_index['preference'])}")
    print(f"\nMemories stored at: {memory_db.persist_dir.absolute()}")
    print(f"Scenarios processed: {len(scenarios)}")

    # Export backup
    backup_path = "memory_backup.json"
    memory_db.export_json(backup_path)
    print(f"\n✓ Backup exported to: {backup_path}")

    print(f"\n{'=' * 70}")

    for mem in seed_memories:
        memory_db.upsert_memory(mem)

    print(f"  ✓ Seeded {len(seed_memories)} memories")

    # Initialize OpenAI client
    print("\n[3/3] Connecting to LLM...")
    openai_api_base = "http://127.0.0.1:25000/v1"
    client = OpenAI(base_url=openai_api_base, api_key="EMPTY")

    # Create agent
    agent = DemandRecognitionAgent(memory_db, client)

    print("\n" + "="*70)
    print("SYSTEM READY")
    print("="*70)

    # Demo scenarios
    scenarios = [
        {
            "name": "Classroom Presentation",
            "description": """Teacher asks user to present climate slide now. User is standing at podium 
            in classroom. User opened PowerPoint file 'climate_presentation.pptx' and is viewing slide 3. 
            Students are seated and waiting. Projector is available."""
        },
        {
            "name": "Library Note Taking",
            "description": """User is in library reading a research paper about AI. User highlighted an 
            important paragraph about neural networks and wants to save it for later review. User's hand 
            is hovering over the highlighted text."""
        },
        {
            "name": "Coffee Shop Meeting",
            "description": """User is sitting in coffee shop with a colleague. Colleague asks 'Can you 
            show me that product demo?' User has laptop open with demo video file. User wants to share 
            screen to colleague's device."""
        }
    ]

    # Run demos
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n\n{'#'*70}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'#'*70}")
        print(f"\nSituation: {scenario['description']}")

        result = agent.analyze_situation(scenario['description'])

        # Print results
        print(f"\n{'='*70}")
        print("ANALYSIS RESULTS")
        print(f"{'='*70}")

        print("\n--- Retrieved Memories ---")
        for mem in result['retrieved_memories']:
            print(f"  • [{mem['memory_id'][:8]}...] (score: {mem['score']:.3f}, sim: {mem['similarity']:.3f})")
            print(f"    {mem['summary'][:80]}...")

        print("\n--- Demand Inference ---")
        demand = result['demand_inference']
        for aspect in ['what', 'when', 'where', 'how']:
            if aspect in demand:
                info = demand[aspect]
                print(f"\n  {aspect.upper()}: {info.get('label', 'N/A')} (confidence: {info.get('confidence', 0):.2f})")
                print(f"    Description: {info.get('description', 'N/A')}")
                evidence = info.get('evidence', [])
                if evidence:
                    print(f"    Evidence: {', '.join(evidence[:2])}")

        print("\n--- Solution Options ---")
        for sol in result['solution_options']:
            print(f"\n  [{sol['rank']}] {sol['description']}")
            print(f"      Confidence: {sol.get('confidence', 0):.2f}")
            print(f"      Reasoning: {sol.get('reasoning', 'N/A')[:100]}...")

            if 'actions' in sol and sol['actions']:
                print(f"      Actions:")
                for action in sol['actions'][:2]:
                    print(f"        - {action.get('type', 'N/A')}: {action.get('payload', 'N/A')[:60]}...")

            if 'ui_ux' in sol:
                ux = sol['ui_ux']
                print(f"      UI/UX: {ux.get('display_method', 'N/A')}, {ux.get('notification_style', 'N/A')}")

        # Save result to file
        output_path = f"demand_recognition_scenario_{i}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Saved detailed result to: {output_path}")

    # Final statistics
    print(f"\n\n{'='*70}")
    print("SESSION SUMMARY")
    print(f"{'='*70}")
    print(f"Total memories in database: {len(memory_db.memories)}")
    print(f"  Episodic: {len(memory_db.metadata_index['episodic'])}")
    print(f"  Semantic: {len(memory_db.metadata_index['semantic'])}")
    print(f"  Preference: {len(memory_db.metadata_index['preference'])}")
    print(f"\nScenarios processed: {len(scenarios)}")
    print(f"\n{'='*70}")


# ---- Interactive Mode ----
def interactive_mode():
    """Interactive demo where user can input custom situations."""

    print("\n" + "="*70)
    print("INTERACTIVE DEMAND RECOGNITION DEMO")
    print("="*70)

    # Initialize system
    memory_db = MemoryDatabase(embedding_dim=384, model_name="all-MiniLM-L6-v2")

    # Seed with example memories
    seed_memories = [
        {
            "type": "preference",
            "timestamp": "2024-01-15T10:30:00Z",
            "source": "user_settings",
            "context": {"scene_label": "classroom"},
            "modalities": {
                "text": "User prefers slides with speaker script visible and voice readout enabled."
            },
            "intent_inference": {
                "what": {"label": "present_slides", "confidence": 0.9},
                "where": {"label": "classroom_projector", "confidence": 0.85},
                "how": {"label": "voice_and_script", "confidence": 0.95}
            },
            "derived_concepts": ["pref_show_script", "pref_voice_readout"],
            "provenance": {"agent": "preference_learner", "confidence_overall": 0.92},
            "privacy": {"sensitivity": "low", "retention_policy": "long_term"}
        },
        {
            "type": "episodic",
            "timestamp": datetime.now().isoformat() + "Z",
            "source": "multimodal_perception",
            "context": {"scene_label": "classroom_front", "relative_location": "standing_at_podium"},
            "modalities": {
                "text": "ASR: 'Please present now.' Opened file: climate_presentation.pptx, slide_index:3",
                "vision": "Teacher standing, students seated, projector screen visible",
                "gesture": "user_hand_gesture_forward"
            },
            "intent_inference": {
                "what": {"label": "show_slide", "confidence": 0.88},
                "when": {"label": "immediate", "confidence": 0.95},
                "where": {"label": "classroom_projector", "confidence": 0.90},
                "how": {"label": "mirror_slide_and_show_script", "confidence": 0.85}
            },
            "derived_concepts": ["presentation_mode", "classroom_context"],
            "provenance": {"agent": "multimodal_reasoner", "confidence_overall": 0.89},
            "privacy": {"sensitivity": "medium", "retention_policy": "standard"}
        }
    ]

    for mem in seed_memories:
        memory_db.upsert_memory(mem)

    print(f"✓ Loaded memory database with {len(seed_memories)} seed memories")

    # Initialize OpenAI client
    openai_api_base = "http://127.0.0.1:25000/v1"
    client = OpenAI(base_url=openai_api_base, api_key="EMPTY")

    # Create agent
    agent = DemandRecognitionAgent(memory_db, client)

    print("\n" + "="*70)
    print("READY - Enter user situations to analyze")
    print("Commands: 'quit' to exit, 'stats' for memory statistics")
    print("="*70)

    while True:
        print("\n" + "-"*70)
        user_input = input("\nDescribe current situation (or command): ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nExiting interactive mode...")
            break

        if user_input.lower() == 'stats':
            print(f"\nMemory Database Statistics:")
            print(f"  Total memories: {len(memory_db.memories)}")
            print(f"  Episodic: {len(memory_db.metadata_index['episodic'])}")
            print(f"  Semantic: {len(memory_db.metadata_index['semantic'])}")
            print(f"  Preference: {len(memory_db.metadata_index['preference'])}")
            continue

        # Analyze situation
        try:
            result = agent.analyze_situation(user_input)

            # Display results
            print("\n" + "="*70)
            print("ANALYSIS COMPLETE")
            print("="*70)

            print("\n📋 RETRIEVED MEMORIES:")
            for mem in result['retrieved_memories'][:3]:
                print(f"  • Score: {mem['score']:.3f} | {mem['summary'][:70]}...")

            print("\n🎯 DEMAND INFERENCE:")
            demand = result['demand_inference']
            for aspect in ['what', 'when', 'where', 'how']:
                if aspect in demand:
                    info = demand[aspect]
                    conf = info.get('confidence', 0)
                    conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
                    print(f"  {aspect.upper()}: {info.get('label', 'N/A')}")
                    print(f"    Confidence: [{conf_bar}] {conf:.2f}")

            print("\n💡 SOLUTION OPTIONS:")
            for sol in result['solution_options']:
                print(f"\n  [{sol['rank']}] {sol['description']}")
                print(f"      Confidence: {sol.get('confidence', 0):.2f}")

                if 'actions' in sol and sol['actions']:
                    print(f"      Actions: {len(sol['actions'])} step(s)")
                    for action in sol['actions'][:1]:
                        print(f"        → {action.get('type', 'N/A')}: {action.get('payload', 'N/A')[:50]}...")

            # Ask if user wants to save
            save = input("\nSave this analysis? (y/n): ").strip().lower()
            if save == 'y':
                # Convert numpy types to native Python types for JSON serialization
                def convert_to_serializable(obj):
                    """Recursively convert numpy types to Python native types."""
                    import numpy as np
                    
                    if isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(item) for item in obj]
                    elif isinstance(obj, (np.floating, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.integer, np.int32, np.int64)):
                        return int(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return obj
                
                # Convert result before saving
                serializable_result = convert_to_serializable(result)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"demand_analysis_{timestamp}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(serializable_result, f, indent=2, ensure_ascii=False)
                print(f"✓ Saved to {filename}")

        except Exception as e:
            print(f"\n✗ Error during analysis: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("SESSION ENDED")
    print("="*70)

# ---- Main Entry Point ----
def main():
    """Main entry point with mode selection."""
    import sys

    print("\n" + "="*70)
    print("DEMAND RECOGNITION AGENT WITH MEMORY RETRIEVAL")
    print("="*70)
    print("\nModes:")
    print("  1. Demo mode - Run predefined scenarios")
    print("  2. Interactive mode - Enter custom situations")
    print("  3. Exit")

    mode = input("\nSelect mode (1/2/3): ").strip()

    if mode == '1':
        demo_demand_recognition()
    elif mode == '2':
        interactive_mode()
    elif mode == '3':
        print("\nExiting...")
        sys.exit(0)
    else:
        print("\nInvalid selection. Exiting...")
        sys.exit(1)


if __name__ == "__main__":
    main()