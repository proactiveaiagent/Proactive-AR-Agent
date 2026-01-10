from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from ..memory.chroma_memory import ChromaUserMemory
from ..models.base import MultimodalInput, MultimodalModel
from ..schema.scene_schema import InteractiveObject, StoredObject, ObjectMatchResult
from ..schema.scene_schema import Face, PersonMatchResult, StoredPerson
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


class PersonMatcher:
    """
    处理检测到的人物与存储的人物信息匹配
    """
    
    def __init__(self, *, model: MultimodalModel, memory: ChromaUserMemory):
        self.model = model
        self.memory = memory
    
    def process_detected_people(
        self,
        *,
        user_id: str,
        detected_people: list[PersonItem],
        faces: Optional[list[Face]] = None  # people_id -> face_embedding
    ) -> list[PersonMatchResult]:
        """
        处理检测到的人物列表，与存储的人物进行匹配
        
        Args:
            user_id: 用户ID
            detected_people: 当前场景检测到的人物列表
            face_embeddings: 每个人物的人脸特征向量 {people_id: embedding}
            
        Returns:
            匹配结果列表
        """
        if not faces:
            return []
        
        results = []
        people_sorted = sorted(detected_people, key=lambda p: p.people_id)
        for i, person in enumerate(people_sorted):
            # 获取该人物的人脸特征向量
            if i >= len(faces):
                continue
            
            face_emb = faces[i].face_emb
            # 使用人脸向量检索相似人物
            similar_people = self.memory.query_people_by_face(
                user_id=user_id,
                face_embedding=face_emb,
                k=1,
                similarity_threshold=0.43
            )
            
            if similar_people:
                # 找到匹配的人物
                best_match = similar_people[0]
                stored_data = best_match["person_data"]
                
                # 更新见面次数和时间
                stored_data["seen_count"] = stored_data.get("seen_count", 0) + 1
                stored_data["last_seen"] = datetime.now(timezone.utc).isoformat()
                
                # 更新存储
                self.memory.upsert_person(
                    user_id=user_id,
                    person_id=best_match["person_id"],
                    person_data=stored_data,
                    face_embedding=face_emb
                )
                
                results.append(PersonMatchResult(
                    person_id=best_match["person_id"],
                    is_match=True,
                    confidence=best_match["similarity"],
                    reasoning=f"Face similarity: {best_match['similarity']:.2f}",
                    stored_person=StoredPerson(**stored_data)
                ))
            else:
                # 创建新人物记录
                new_person_id = f"person_{user_id}_{datetime.now(timezone.utc).timestamp()}"
                new_person_data = {
                    "person_id": new_person_id,
                    "name": None,
                    "role": person.role,
                    "kinship_term": person.kinship_term,
                    "relationship_notes": "",
                    "first_seen": datetime.now(timezone.utc).isoformat(),
                    "last_seen": datetime.now(timezone.utc).isoformat(),
                    "seen_count": 1,
                    "interaction_history": [],
                    "typical_locations": []
                }
                
                self.memory.upsert_person(
                    user_id=user_id,
                    person_id=new_person_id,
                    person_data=new_person_data,
                    face_embedding=face_emb
                )
                
                results.append(PersonMatchResult(
                    person_id=new_person_id,
                    is_match=False,
                    confidence=0.0,
                    reasoning="New person detected",
                    stored_person=StoredPerson(**new_person_data)
                ))
        
        return results