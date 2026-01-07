from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import ValidationError

from ..memory.chroma_memory import ChromaUserMemory
from ..models.base import MultimodalInput, MultimodalModel
from ..schema.scene_schema import SceneDescription
from .prompts import SCENE_JSON_SYSTEM_PROMPT, SCENE_JSON_USER_PROMPT
from .ar_matcher import ObjectMatcher


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

    def run(self, *, user_id: str, mm_input: MultimodalInput) -> SceneDescription:
        memory_snippets = self.memory.query(user_id=user_id, query="用户环境/物体/行为偏好", k=self.cfg.memory_query_k)
        user_prompt = self._build_user_prompt(user_id=user_id, memory_snippets=memory_snippets)
        if mm_input is None:
            mm_input = MultimodalInput()
        raw = self.model.generate(
            system_prompt=SCENE_JSON_SYSTEM_PROMPT,
            user_prompt=user_prompt,
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
            scene = SceneDescription.model_validate(data)
        except ValidationError as e:
            # Bubble up with raw output for debugging.
            raise ValueError(f"Scene JSON validation failed: {e}\nRaw: {raw}") from e

        # Upsert memory: store a short embedding text + full payload JSON.
        if scene.interactive_objects_detail:
            matched_objects = self.object_matcher.process_detected_objects(
                user_id=user_id,
                detected_objects=scene.interactive_objects_detail
            )
            scene.stored_objects = matched_objects
        return scene
