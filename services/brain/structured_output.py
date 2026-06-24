from __future__ import annotations

from pydantic import BaseModel

from services.april_runtime.schemas import ResponseFormat
from services.brain.schemas import BrainDecision


def response_format_for_model(model: type[BaseModel]) -> ResponseFormat:
    """Build a JSON-object response format from a Pydantic model's own schema.

    Deriving the schema from the model avoids maintaining a duplicated, drifting
    copy: the structured-output constraint always tracks the validated type.
    """
    return ResponseFormat(type="json_object", json_schema=model.model_json_schema())


# Computed once at import; the brain asks the runtime to constrain output to the
# exact routing schema it will then validate against.
BRAIN_DECISION_RESPONSE_FORMAT = response_format_for_model(BrainDecision)
