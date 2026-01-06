from __future__ import annotations

from ..schema.scene_schema import SceneDescription, ObjectMatchResult
from ..schema.schema_prompt_builder import generate_prompt_from_model


# Generate dynamic schema description
SCHEMA_DESCRIPTION = generate_prompt_from_model(SceneDescription)
OBJECT_MATCH_RESULT_PROMPT = generate_prompt_from_model(ObjectMatchResult)

SCENE_JSON_SYSTEM_PROMPT = """You are a multimodal scene understanding assistant for AR.
Return ONLY valid JSON (no markdown, no code fences, no extra commentary).
All fields must be present. If something is not visible/audible, use empty list, null.
"""

SCENE_JSON_USER_PROMPT = f"""Analyze the provided video frames (and optional ASR/audio hints) and produce a JSON object that matches the required schema.

Constraints:
- Output MUST be a single JSON object.
- Use concise but complete descriptions.
- Follow the exact schema structure below.

{SCHEMA_DESCRIPTION}

Task:
Analyze the scene and fill in all required fields based on your observations.
"""

OBJECT_MATCH_SYSTEM_PROMPT = OBJECT_MATCH_SYSTEM_PROMPT = f"""You are a professional object matching expert.
Your task is to determine whether two object descriptions refer to the same physical object. 
Judgment criteria:
1. The types must be identical or highly related.
2. The descriptions of key features (type, affordance) should be similar.
3. Consider the possible differences in descriptions due to factors such as lighting and angle.
4. Output in JSON format: {OBJECT_MATCH_RESULT_PROMPT}
"""

OBJECT_MATCH_USER_PROMPT = """Please determine whether the following two objects are the same. 
**New Detected Object:**
Name: {new_object_name}
Description: {new_description} 
**Objects in Memory:**
Name: {stored_object_name}
Description: {stored_description} 
Please output the matching results in JSON format.
"""
