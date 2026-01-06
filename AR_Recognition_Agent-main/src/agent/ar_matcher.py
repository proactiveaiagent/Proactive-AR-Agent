from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Optional

from ..memory.chroma_memory import ChromaUserMemory
from ..models.base import MultimodalInput, MultimodalModel
from ..schema.scene_schema import InteractiveObject, StoredObject, ObjectMatchResult
from ..schema.object_description_builder import object_to_description
from .prompts import OBJECT_MATCH_SYSTEM_PROMPT, OBJECT_MATCH_USER_PROMPT


class ObjectMatcher:

    MATCH_CONFIDENCE_THRESHOLD = 0.5
    
    def __init__(
        self,
        model: MultimodalModel,
        memory: ChromaUserMemory,
    ):
        self.model = model
        self.memory = memory
    
    def process_detected_objects(
        self,
        user_id: str,
        detected_objects: list[InteractiveObject]
    ) -> list[StoredObject]:
        results = []
        
        for obj_data in detected_objects:
            matched_obj = self._match_or_create_object(user_id, obj_data)
            results.append(matched_obj)
        
        return results
    
    def _match_or_create_object(
        self,
        user_id: str,
        detected: InteractiveObject
    ) -> StoredObject:
        
        description_candidates = self._query_by_name_and_description(
            user_id,
            detected.object_name,
            object_to_description(detected),
        )
        
        # Step 4: 使用大模型判断是否为同一物体
        for candidate in description_candidates:
            match_result = self._llm_match_objects(detected, candidate)
            
            if match_result.is_same and match_result.confidence >= self.MATCH_CONFIDENCE_THRESHOLD:
                # 找到匹配，更新已有物体 - 传入新检测到的 InteractiveObject
                return self._update_existing_object(user_id, candidate, detected)
        
        # Step 5: 没有匹配到，创建新物体
        return self._create_new_object(user_id, detected)
    
    def _query_by_name_and_description(
        self,
        user_id: str,
        object_name: str,
        description: str,
        k: int = 3
    ) -> list[StoredObject]:
        filter_meta = {"type": "object"}
        
        results = self.memory.query(
            user_id=user_id,
            query=f"object_name: {object_name}; object_description: {description}",
            k=k,
            filter_metadata=filter_meta
        )
        
        candidates = []
        for result in results:
            # 修复：从正确的位置获取 object_data
            event = result.get("event", {})
            obj_data = event.get("object_data", {})
            if obj_data:
                candidates.append(StoredObject(**obj_data))
        
        return candidates
    
    def _llm_match_objects(
        self,
        detected: InteractiveObject,
        stored: StoredObject
    ) -> ObjectMatchResult:
        user_prompt = OBJECT_MATCH_USER_PROMPT.format(
            new_object_name=detected.object_name,
            new_description=object_to_description(detected),
            stored_object_name=stored.object_name,
            stored_description=object_to_description(stored),
        )
        
        raw_output = self.model.generate(
            system_prompt=OBJECT_MATCH_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            mm_input=MultimodalInput(),
        )
        
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            start = raw_output.find("{")
            end = raw_output.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(raw_output[start:end+1])
            else:
                return ObjectMatchResult(
                    is_same=False,
                    confidence=0.0,
                    reasoning="Error parsing LLM output"
                )
        
        return ObjectMatchResult(**data)
    
    def _create_new_object(
        self,
        user_id: str,
        detected: InteractiveObject
    ) -> StoredObject:
        now = datetime.now().isoformat()
        object_id = str(uuid.uuid4())
        
        new_obj = StoredObject(
            object_id=object_id,
            object_name=detected.object_name,
            object_type=detected.object_type,
            spatial_relation=detected.spatial_relation,
            current_state=detected.current_state,
            affordance=detected.affordance,
            digital_connectivity=detected.digital_connectivity,
            first_seen=now,
            last_seen=now,
            seen_count=1
        )
        
        description = object_to_description(detected)
        embedding_text = f"object_name: {new_obj.object_name}; object_description: {description}"
        
        self.memory.upsert_event(
            user_id=user_id,
            event={
                "event_id": object_id,
                "type": "object",
                "object_data": new_obj.model_dump()
            },
            text_for_embedding=embedding_text,
            metadata={
                "type": "object",
                "object_name": new_obj.object_name,
                "object_type": new_obj.object_type,
                "object_id": new_obj.object_id
            }
        )
        
        return new_obj
    
    def _update_existing_object(
        self,
        user_id: str,
        stored: StoredObject,
        detected: InteractiveObject 
    ) -> StoredObject:
        now = datetime.now().isoformat()
        
        stored.object_type = detected.object_type
        stored.spatial_relation = detected.spatial_relation
        stored.current_state = detected.current_state
        stored.affordance = detected.affordance
        stored.digital_connectivity = detected.digital_connectivity
        stored.last_seen = now
        stored.seen_count += 1
        
        new_description = object_to_description(detected)
        embedding_text = f"object_name: {stored.object_name}; object_description: {new_description}"
        
        self.memory.upsert_event(
            user_id=user_id,
            event={
                "event_id": stored.object_id, 
                "type": "object",
                "object_data": stored.model_dump()
            },
            text_for_embedding=embedding_text,
            metadata={
                "type": "object",
                "object_name": stored.object_name,
                "object_type": stored.object_type,
                "object_id": stored.object_id
            }
        )
        
        return stored