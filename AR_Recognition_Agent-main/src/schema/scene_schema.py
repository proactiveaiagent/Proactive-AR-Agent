from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class PersonItem(BaseModel):
    thought: str
    people_id: str
    role: str
    location_relative_to_user: str
    attention_target: str
    activity_state: str
    kinship_term: str


class DetectedPeopleAnalysis(BaseModel):
    relationship_situation_summary: str
    people_list: List[PersonItem] = Field(default_factory=list)


class InteractiveObject(BaseModel):
    thought: str
    object_name: str
    object_type: str
    spatial_relation: str
    current_state: str
    affordance: List[str] = Field(default_factory=list)
    digital_connectivity: str


class DetectedTextItem(BaseModel):
    text_content: str
    text_source_description: str
    text_role: str
    associated_object_id: Optional[str] = None
    is_interactive: bool = False
    ocr_confidence: str


class SpatialEnvironmentalAnalysis(BaseModel):
    user_reach_range: str
    critical_interaction_zone: str
    lighting_state: str
    noise_level_category: str
    safety_hazards: str


class SoundEvent(BaseModel):
    event_type: str
    source_location: str
    sound_level_description: str
    asr_transcript: Optional[str] = None
    asr_confidence: Optional[str] = None


class WithARSystem(BaseModel):
    common_apps: List[str] = Field(default_factory=list)
    typical_behaviors: List[str] = Field(default_factory=list)


class UserInteractions(BaseModel):
    with_surroundings: List[str] = Field(default_factory=list)
    with_ar_system: WithARSystem = Field(default_factory=WithARSystem)
    with_agents: List[str] = Field(default_factory=list)


class UserStatus(BaseModel):
    status_inference: str
    observable_behaviors: List[str] = Field(default_factory=list)
    gaze_target: str
    gaze_duration: str
    peripheral_awareness: List[str] = Field(default_factory=list)


class SceneDescription(BaseModel):
    thought: str
    scene_narrative: str
    location_tag: str
    what_is_happening: str
    spatial_environmental_analysis: SpatialEnvironmentalAnalysis
    detected_text_in_scene: List[DetectedTextItem] = Field(default_factory=list)
        
        
class ObjectDescription(BaseModel):
    interactive_objects_detail: List[InteractiveObject] = Field(default_factory=list)
    
    
class PersonDescription(BaseModel):
    detected_people_analysis: DetectedPeopleAnalysis
    user_status: UserStatus
    user_interactions: UserInteractions
    
    # Audio
    is_user_speaking: Optional[bool] = None
    sound_events_detected: List[SoundEvent] = Field(default_factory=list)
        
        
class FinalDescription(BaseModel):
    scene: SceneDescription
    objects: ObjectDescription
    people: PersonDescription

    # Allow extra model-specific fields if needed.
    extra: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class StoredObject(BaseModel):
    object_id: str
    object_name: str
    object_type: str
    spatial_relation: str
    current_state: str
    affordance: List[str] = Field(default_factory=list)
    digital_connectivity: str
    first_seen: str
    last_seen: str
    seen_count: int

class ObjectMatchResult(BaseModel):
    is_same: bool
    confidence: float
    reasoning: str

class StoredPerson(BaseModel):
    person_id: str
    name: Optional[str] = None
    role: str
    kinship_term: str
    relationship_notes: str = ""
    first_seen: str
    last_seen: str
    seen_count: int
    interaction_history: List[str] = Field(default_factory=list)
    typical_locations: List[str] = Field(default_factory=list)
    
class PersonMatchResult(BaseModel):
    person_id: str
    is_match: bool
    confidence: float
    reasoning: str
    stored_person: Optional[StoredPerson] = None
    

class Face:
    def __init__(self, bounding_box, face_emb, cluster_id, extra_data):
        self.bounding_box = bounding_box
        self.face_emb = face_emb
        self.cluster_id = cluster_id
        self.extra_data = extra_data