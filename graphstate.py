from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ConversationItem(BaseModel):
    query: str = ""
    response: str = ""


class ShopState(BaseModel):
    # Required
    query: str = Field(default="", description="User query")

    # Optional
    phone: Optional[str] = ""
    history: Optional[List[ConversationItem]] = None

    # Internal state
    sentiment: Optional[str] = None
    departments: Optional[List[str]] = None
    responses: Optional[List[str]] = None
    final_response: Optional[str] = None
    escalation_required: Optional[bool] = False
    rag_docs_found: Optional[bool] = None
    rewritten_query: Optional[str] = None
    retry_count: Optional[int] = 0

    out_of_scope: Optional[bool] = False
    guardrail_reason: Optional[str] = None

    rewritten_query: Optional[str] = None
    needs_rewrite: Optional[bool] = False

    optimized_query: Optional[str] = None

    turn_type: Optional[str] = None

    is_satisfactory: Optional[bool] = None
    faithfulness_score: Optional[int] = 0
    relevance_score: int = 0
    improvement_feedback: Optional[str] = ""

    node_name: Optional[str] = None

    

    @field_validator("history", mode="before")
    def default_history(cls, v):
        return v or []

    @field_validator("departments", mode="before")
    def default_departments(cls, v):
        return v or []

    @field_validator("responses", mode="before")
    def default_responses(cls, v):
        return v or []
