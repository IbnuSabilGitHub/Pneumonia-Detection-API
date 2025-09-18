from typing import Any, Dict, Optional


def build_response(
    description: str,
    model: Optional[type] = None,
    *,
    examples: Dict[str, Dict[str, Any]] | None = None,
    example: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build a FastAPI 'responses' entry with optional models and examples.

    Rules:
    - Use 'examples' for multiple named examples.
    - Use 'example' for a single, simple example.
    """
    block: Dict[str, Any] = {"description": description}
    if model:
        block["model"] = model
    content: Dict[str, Any] = {}
    if examples:
        content = {"application/json": {"examples": examples}}
    elif example:
        content = {"application/json": {"example": example}}
    if content:
        block["content"] = content
    return block
