from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import ValidationError

from ..memory.chroma_memory import ChromaUserMemory
from ..models.base import MultimodalInput, MultimodalModel
from ..schema.scene_schema import SceneDescription, ObjectDescription, PersonDescription, FinalDescription
from .prompts import SCENE_JSON_SYSTEM_PROMPT, SCENE_JSON_USER_PROMPT, OBJECT_JSON_SYSTEM_PROMPT, OBJECT_JSON_USER_PROMPT, PERSON_JSON_SYSTEM_PROMPT, PERSON_JSON_USER_PROMPT
from .ar_matcher import ObjectMatcher, PersonMatcher


@dataclass
class AgentConfig:
    memory_query_k: int = 6


class ARRecognitionAgent:
    def __init__(
        self,
        *,
        model: MultimodalModel,
        memory: ChromaUserMemory,
        cfg: Optional[AgentConfig] = None,
    ):
        self.model = model
        self.memory = memory
        self.cfg = cfg or AgentConfig()
        self.object_matcher = ObjectMatcher(model=model, memory=memory)
        self.person_matcher = PersonMatcher(model=model, memory=memory)

    def _build_user_prompt(self, *, user_id: str, memory_snippets: list[dict[str, Any]]) -> str:
        # Feed recent relevant memory back into the prompt.
        memory_text = json.dumps(memory_snippets, ensure_ascii=False)
        return (
            SCENE_JSON_USER_PROMPT
            + "\n\nUserId: "
            + user_id
            + "\nRelevant memory (JSON list):\n"
            + memory_text
        )
    
    def _build_object_user_prompt(self, *, user_id: str, memory_snippets: list[dict[str, Any]]) -> str:
        # Feed recent relevant memory back into the prompt.
        memory_text = json.dumps(memory_snippets, ensure_ascii=False)
        return (
            OBJECT_JSON_USER_PROMPT
            + "\n\nUserId: "
            + user_id
            + "\nRelevant memory (JSON list):\n"
            + memory_text
        )
        
    def _build_person_user_prompt(self, *, user_id: str, memory_snippets: list[dict[str, Any]]) -> str:
        # Feed recent relevant memory back into the prompt.
        memory_text = json.dumps(memory_snippets, ensure_ascii=False)
        return (
            PERSON_JSON_USER_PROMPT
            + "\n\nUserId: "
            + user_id
            + "\nRelevant memory (JSON list):\n"
            + memory_text
        )

    def run(self, *, user_id: str, mm_input: MultimodalInput) -> SceneDescription:
        memory_snippets = self.memory.query(user_id=user_id, query="用户环境/物体/行为偏好", k=self.cfg.memory_query_k)
        scene_prompt = self._build_user_prompt(user_id=user_id, memory_snippets=memory_snippets)
        object_prompt = self._build_object_user_prompt(user_id=user_id, memory_snippets=memory_snippets)
        person_prompt = self._build_person_user_prompt(user_id=user_id, memory_snippets=memory_snippets)
        system_prompts = [SCENE_JSON_SYSTEM_PROMPT, OBJECT_JSON_SYSTEM_PROMPT, PERSON_JSON_SYSTEM_PROMPT]
        prompts = [scene_prompt, object_prompt, person_prompt]
        prompts_type = ['scene', 'objects', 'person']
        answers_schema = [SceneDescription, ObjectDescription, PersonDescription]
        answers = {}
        if mm_input is None:
            mm_input = MultimodalInput()
            
        for i, prompt in enumerate(prompts):
            
            if prompts_type[i] == 'person':
                mm_input.metadata = mm_input.metadata or {}
                mm_input.metadata['use_ann_img'] = True
                
            raw = self.model.generate(
                system_prompt=system_prompts[i],
                user_prompt=prompt,
                mm_input=mm_input,
            )

            # Parse JSON robustly.
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Fallback: attempt to extract a JSON object substring.
                start = raw.find("{")
                end = raw.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise
                data = json.loads(raw[start : end + 1])

            try:
                ans = answers_schema[i].model_validate(data)
                answers[prompts_type[i]] = ans
            except ValidationError as e:
                # Bubble up with raw output for debugging.
                raise ValueError(f"Scene JSON validation failed: {e}\nRaw: {raw}") from e
            
        final_description = FinalDescription(
            scene=answers['scene'],
            objects=answers['objects'],
            people=answers['person'],
        )
        
        # Upsert memory: store a short embedding text + full payload JSON.
        if final_description.objects.interactive_objects_detail:
            matched_objects = self.object_matcher.process_detected_objects(
                user_id=user_id,
                detected_objects=final_description.objects.interactive_objects_detail
            )
            final_description.stored_objects = matched_objects
        
        # 处理人物匹配
        if final_description.people.detected_people_analysis.people_list:
            # 从 mm_input 中获取人脸特征向量
            faces = mm_input.faces if mm_input and mm_input.faces else None
            
            matched_people = self.person_matcher.process_detected_people(
                user_id=user_id,
                detected_people=final_description.people.detected_people_analysis.people_list,
                faces=faces
            )
            final_description.matched_people = matched_people
        
        return final_description
