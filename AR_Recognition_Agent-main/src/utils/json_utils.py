from __future__ import annotations

import json
from typing import Any


def load_first_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object found in a string."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])
