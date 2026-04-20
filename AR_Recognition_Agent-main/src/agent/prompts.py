from __future__ import annotations

from ..schema.scene_schema import SceneDescription, ObjectDescription, PersonDescription, ObjectMatchResult
from ..schema.schema_prompt_builder import generate_prompt_from_model


# Generate dynamic schema description
SCENE_DESCRIPTION = generate_prompt_from_model(SceneDescription)
OBJECT_DESCRIPTION_PROMPT = generate_prompt_from_model(ObjectDescription)
PERSON_DESCRIPTION_PROMPT = generate_prompt_from_model(PersonDescription)
OBJECT_MATCH_RESULT_PROMPT = generate_prompt_from_model(ObjectMatchResult)

SCENE_JSON_SYSTEM_PROMPT = """You are a multimodal scene understanding assistant for AR.
Return ONLY valid JSON (no markdown, no code fences, no extra commentary).
All fields must be present. If something is not visible/audible, use empty list, null.
"""

SCENE_JSON_USER_PROMPT = f"""Analyze the provided video frames (and optional ASR/audio hints) and produce a JSON object that matches the required schema.

Input Information:
- The perspective of this video is that of the user himself/herself.
- The audio transcription contains all the spoken words in the scene.

Constraints:
- Output MUST be a single JSON object.
- Use concise but complete descriptions.
- Follow the exact schema structure below.
- Output language should be English.

Output Requirements:
{SCENE_DESCRIPTION}
"""

OBJECT_JSON_SYSTEM_PROMPT = """You are a professional object detection expert.
Your task is to generate detailed descriptions of objects detected in the scene based on visual and audio cues.
Return ONLY valid JSON (no markdown, no code fences, no extra commentary).
"""

OBJECT_JSON_USER_PROMPT = f"""Please generate detailed descriptions for the detected objects in the scene based on the provided visual and audio information. The object type includes but is not limited to food, appliances, personal items, and other tangible items.

Input Information:
- The perspective of this video is that of the user himself/herself.
- The audio transcription contains all the spoken words in the scene.

Constraints:
- Focus on observable objects' attributes and behaviors.
- Use concise but complete descriptions.
- Follow the exact schema structure below.
- Output language should be English.

Output Requirements:
{OBJECT_DESCRIPTION_PROMPT}
"""

PERSON_JSON_SYSTEM_PROMPT = """You are a professional person description expert.
Your task is to generate detailed descriptions of people detected in the scene based on visual and audio cues.
Return ONLY valid JSON (no markdown, no code fences, no extra commentary).
"""

PERSON_JSON_USER_PROMPT = f"""Please generate detailed descriptions for the detected people in the scene based on the provided visual and audio information.

Input Information:
- The perspective of the video is exactly the perspective that the user's eyes would see.
- The audio transcription contains all the spoken words in the scene.
- The face will be labeled with a unique people_id that can be used to link to face embeddings.

Constraints:
- Focus on observable attributes and behaviors.
- Use concise but complete descriptions.
- Follow the exact schema structure below.
- The Id of each person should match the index id labeled on the images.
- If there is no person detected, return empty fields/lists accordingly.
- Output language should be English. Except for the audio transcription which may contain other languages.

Output Requirements:
{PERSON_DESCRIPTION_PROMPT}
"""

OBJECT_MATCH_SYSTEM_PROMPT = f"""You are a professional object matching expert.
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
