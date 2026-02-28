from typing import List, Optional, Literal, Annotated, Dict
# Note: We don't need operator anymore for the replacement reducer
from pydantic import BaseModel, Field

class ConversationItem(BaseModel):
    query: str = ""
    response: str = ""

class AffectiveOutput(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    emotion: Literal[
        "anger", "frustration", "confusion", "fear", "sadness", 
        "disappointment", "neutral", "satisfaction", "gratitude"
    ]
    emotion_intensity: float
    requires_escalation: bool

# Define a simple "replace" reducer
def replace_value(old, new):
    return new

class ShopState(BaseModel):
    # ==========================================================
    # ANNOTATED FIELDS (Fixed Reducer)
    # ==========================================================
    
    # We use a lambda or the replace_value function to take (current, new) -> new
    node_name: Annotated[str, replace_value] = ""
    query: Annotated[str, replace_value] = ""
    optimized_query: Annotated[Optional[str], replace_value] = None
    turn_type: Annotated[Optional[str], replace_value] = None
    escalation_required: Annotated[Optional[bool], replace_value] = False
    
    # ==========================================================
    # STANDARD FIELDS (No conflicts expected)
    # ==========================================================
    phone: Optional[str] = ""
    history: Optional[List[ConversationItem]] = None
    has_attachment: Optional[bool] = False
    departments: Optional[List[str]] = None
    responses: Optional[List[str]] = None
    final_response: Optional[str] = None
    rag_docs_found: Optional[bool] = None
    retry_count: Optional[int] = 0
    out_of_scope: Optional[bool] = False
    guardrail_reason: Optional[str] = None
    skip_guardrail: Optional[bool] = False
    needs_rewrite: Optional[bool] = False
    rewritten_query: Optional[str] = None
    is_satisfactory: Optional[bool] = None
    faithfulness_score: Optional[int] = 0
    relevance_score: int = 0
    improvement_feedback: Optional[str] = ""
    sentiment: Optional[str] = None
    emotion: Optional[str] = None
    emotion_intensity: Optional[float] = None
    topic_shift_detected: Optional[bool] = False
    awaiting_clarification: Optional[bool] = False
    clarification_context: Optional[dict] = None
    clarification_resolved: Optional[str] = None