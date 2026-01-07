from __future__ import annotations

import inspect
from typing import get_args, get_origin, Any, Optional, List, Union

from pydantic import BaseModel


def is_pydantic_model(tp: Any) -> bool:
    return inspect.isclass(tp) and issubclass(tp, BaseModel)


def is_list_type(tp: Any) -> bool:
    return get_origin(tp) in (list, List)


def is_optional_type(tp: Any) -> bool:
    return get_origin(tp) is Union and type(None) in get_args(tp)


def unwrap_optional(tp: Any) -> Any:
    return next(t for t in get_args(tp) if t is not type(None))


def type_to_text(tp: Any) -> str:
    if tp is str:
        return "string"
    if tp is bool:
        return "boolean"
    if tp is int:
        return "integer"
    if tp is float:
        return "number"
    return tp.__name__


def generate_prompt_from_model(
    model: type[BaseModel],
    indent: int = 0,
    is_root: bool = False
) -> str:
    lines = []
    prefix = " " * indent

    # ❗ 模型名只作为注释说明
    lines.append(f"{prefix}# Fields of {model.__name__}")

    for field_name, field_info in model.model_fields.items():
        
        if field_name in {"extra_attributes", "model_extra"}:
            continue
        annotation = field_info.annotation
        is_optional = is_optional_type(annotation)

        if is_optional:
            annotation = unwrap_optional(annotation)

        # List
        if is_list_type(annotation):
            inner = get_args(annotation)[0]
            lines.append(f"\n{prefix}{field_name}: list")

            if is_pydantic_model(inner):
                lines.append(f"{prefix}  # Each item:")
                lines.append(
                    generate_prompt_from_model(
                        inner, indent + 2
                    )
                )
            else:
                lines.append(
                    f"{prefix}  # item type: {type_to_text(inner)}"
                )

        # Nested model
        elif is_pydantic_model(annotation):
            lines.append(f"\n{prefix}{field_name}: object")
            lines.append(
                generate_prompt_from_model(
                    annotation, indent + 2
                )
            )

        else:
            desc = type_to_text(annotation)
            if is_optional:
                desc += " (optional, can be null)"
            lines.append(f"{prefix}{field_name}: {desc}")

    return "\n".join(lines)

