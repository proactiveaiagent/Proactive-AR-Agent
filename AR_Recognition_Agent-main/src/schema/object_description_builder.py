from __future__ import annotations

from typing import Union

from ..schema.scene_schema import InteractiveObject, StoredObject


class ObjectDescriptor:
    
    @staticmethod
    def to_description(obj: Union[InteractiveObject, StoredObject]) -> str:
        parts = []
        
        # Object type
        if obj.object_type:
            parts.append(f"Type: {obj.object_type}")
        
        # Spatial relation
        if obj.spatial_relation:
            parts.append(f"Location: {obj.spatial_relation}")
        
        # Current state
        if obj.current_state:
            parts.append(f"State: {obj.current_state}")
        
        # Affordance (what you can do with it)
        if obj.affordance:
            affordance_str = ", ".join(obj.affordance)
            parts.append(f"Actions available: {affordance_str}")
        
        # Digital connectivity
        if obj.digital_connectivity:
            parts.append(f"Connectivity: {obj.digital_connectivity}")

        
        return "; ".join(parts)
    
    @staticmethod
    def to_compact_description(obj: Union[InteractiveObject, StoredObject]) -> str:

        parts = [obj.object_type]
        
        if obj.spatial_relation:
            parts.append(obj.spatial_relation)
        
        if obj.current_state:
            parts.append(obj.current_state)
        
        if obj.affordance:
            parts.extend(obj.affordance[:2])  # 只取前2个affordance
        
        return " ".join(parts)
    
    @staticmethod
    def to_structured_description(obj: Union[InteractiveObject, StoredObject]) -> str:
        lines = []
        
        lines.append(f"Object Type: {obj.object_type}")
        lines.append(f"Spatial Relation: {obj.spatial_relation}")
        lines.append(f"Current State: {obj.current_state}")
        
        if obj.affordance:
            lines.append(f"Affordances: {', '.join(obj.affordance)}")
        else:
            lines.append("Affordances: none")
        
        lines.append(f"Digital Connectivity: {obj.digital_connectivity}")
        
        
        return "\n".join(lines)
    
    @staticmethod
    def compare_description(obj1: Union[InteractiveObject, StoredObject], 
                           obj2: Union[InteractiveObject, StoredObject]) -> str:
        comparison = []
        
        comparison.append("=== Object 1 ===")
        comparison.append(ObjectDescriptor.to_structured_description(obj1))
        comparison.append("\n=== Object 2 ===")
        comparison.append(ObjectDescriptor.to_structured_description(obj2))
        
        return "\n".join(comparison)


def object_to_description(obj: Union[InteractiveObject, StoredObject]) -> str:
    return ObjectDescriptor.to_description(obj)


def object_to_compact(obj: Union[InteractiveObject, StoredObject]) -> str:
    return ObjectDescriptor.to_compact_description(obj)


def object_to_structured(obj: Union[InteractiveObject, StoredObject]) -> str:
    return ObjectDescriptor.to_structured_description(obj)