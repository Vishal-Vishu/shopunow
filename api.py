from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from graphbuilder import build_graph
from graphstate import ShopState
from multimodalprocessor import process_uploaded_file

from langgraph.graph import StateGraph, END

from graphnodes import (rewrite_node, guardrail_node, guardrail_block_node, department_node,
                        department_execution_node, response_enrichment_node, clarification_node, sentiment_node) 


# ==========================================================
# Initialize FastAPI
# ==========================================================

app = FastAPI(
    title="ShopUNow Agentic AI API",
    version="2.0.0"
)

graph = build_graph()


# ==========================================================
# Request / Response Schemas
# ==========================================================

class ConversationItem(BaseModel):
    query: str
    response: str


class ChatRequest(BaseModel):
    query: str
    phone: Optional[str] = ""
    history: Optional[List[ConversationItem]] = []
    has_attachment: Optional[bool] = False


class ChatResponse(BaseModel):
    response: str
    departments: Optional[List[str]] = None
    sentiment: Optional[str] = None
    emotion: Optional[str] = None
    escalation_required: Optional[bool] = None
    faithfulness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    topic_shift_detected: Optional[bool] = None
    awaiting_clarification: Optional[bool] = None

def build_rewrite_only_graph():
    graph = StateGraph(ShopState)

    graph.add_node("rewrite", rewrite_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", END)

    return graph.compile()

class RewriteTestResponse(BaseModel):
    original_query: str
    optimized_query: Optional[str] = None
    needs_rewrite: Optional[bool] = None
    awaiting_clarification: Optional[bool] = None

def build_guardrail_graph():
    graph = StateGraph(ShopState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("guardrail_block", guardrail_block_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "guardrail")

    graph.add_conditional_edges(
        "guardrail",
        lambda state: "guardrail_block" if state.out_of_scope else END,
        {
            "guardrail_block": "guardrail_block",
            END: END
        }
    )

    graph.add_edge("guardrail_block", END)

    return graph.compile()    

def build_routing_graph():
    graph = StateGraph(ShopState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("department", department_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "guardrail")

    graph.add_conditional_edges(
        "guardrail",
        lambda state: "department" if not state.out_of_scope else END,
        {
            "department": "department",
            END: END
        }
    )

    graph.add_edge("department", END)

    return graph.compile()

def build_execution_graph():
    graph = StateGraph(ShopState)

    graph.add_node("execute_departments", department_execution_node)
    graph.add_node("response_enrichment", response_enrichment_node)

    graph.set_entry_point("execute_departments")
    graph.add_edge("execute_departments", "response_enrichment")
    graph.add_edge("response_enrichment", END)

    return graph.compile()

def build_continue_graph():

    graph = StateGraph(ShopState)

    graph.add_node("clarification_resolver", clarification_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("department", department_node)
    graph.add_node("execute_departments", department_execution_node)
    graph.add_node("response_enrichment", response_enrichment_node)

    graph.set_entry_point("clarification_resolver")

    graph.add_edge("clarification_resolver", "rewrite")
    graph.add_edge("rewrite", "guardrail")
    graph.add_edge("guardrail", "sentiment")
    graph.add_edge("sentiment", "department")
    graph.add_edge("department", "execute_departments")
    graph.add_edge("execute_departments", "response_enrichment")
    graph.add_edge("response_enrichment", END)

    return graph.compile()

# ==========================================================
# FULL AGENT EXECUTION (ASYNC SAFE)
# ==========================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):

    result = await graph.ainvoke({
        "query": request.query,
        "phone": request.phone,
        "history": request.history,
        "has_attachment": request.has_attachment
    })

    return ChatResponse(
        response=result.get("final_response", ""),
        departments=result.get("departments"),
        sentiment=result.get("sentiment"),
        emotion=result.get("emotion"),
        escalation_required=result.get("escalation_required"),
        faithfulness_score=result.get("faithfulness_score"),
        relevance_score=result.get("relevance_score"),
        topic_shift_detected=result.get("topic_shift_detected"),
        awaiting_clarification=result.get("awaiting_clarification")
    )


# ==========================================================
# MULTIMODAL ENDPOINT (FILE SUPPORT)
# ==========================================================

@app.post("/chat-with-file", response_model=ChatResponse)
async def chat_with_file(
    query: str = Form(...),
    phone: str = Form(""),
    file: UploadFile = File(None)
):

    combined_query = query
    has_attachment = False

    if file:
        extracted_text = process_uploaded_file(file.file)
        has_attachment = True

        combined_query = f"""
User Query:
{query}

Attached Document Content:
{extracted_text}
"""

    result = await graph.ainvoke({
        "query": combined_query,
        "phone": phone,
        "history": [],
        "has_attachment": has_attachment
    })

    return ChatResponse(
        response=result.get("final_response", ""),
        departments=result.get("departments"),
        sentiment=result.get("sentiment"),
        emotion=result.get("emotion"),
        escalation_required=result.get("escalation_required"),
        faithfulness_score=result.get("faithfulness_score"),
        relevance_score=result.get("relevance_score"),
        topic_shift_detected=result.get("topic_shift_detected"),
        awaiting_clarification=result.get("awaiting_clarification")
    )


# ==========================================================
# SENTIMENT ONLY
# ==========================================================

class SentimentResponse(BaseModel):
    sentiment: Optional[str]
    emotion: Optional[str]
    emotion_intensity: Optional[float]
    escalation_required: Optional[bool]


@app.post("/analyze-affective", response_model=SentimentResponse)
async def analyze_affective(request: ChatRequest):

    result = await graph.ainvoke({
        "query": request.query,
        "history": request.history or [],
    })

    return SentimentResponse(
        sentiment=result.get("sentiment"),
        emotion=result.get("emotion"),
        emotion_intensity=result.get("emotion_intensity"),
        escalation_required=result.get("escalation_required")
    )


# ==========================================================
# CLARIFICATION CONTINUATION ENDPOINT
# ==========================================================

@app.post("/continue")
async def continue_conversation(request: ChatRequest):
    """
    Used when awaiting_clarification = True
    """

    graph = build_continue_graph()
    result = await graph.ainvoke({
    "query": request.query,
    "phone": request.phone,
    "history": [item.model_dump() for item in request.history] if request.history else [],
    "has_attachment": request.has_attachment
    })

    return {
        "response": result.get("final_response"),
        "awaiting_clarification": result.get("awaiting_clarification")
    }


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/test-rewrite", response_model=RewriteTestResponse)
async def test_rewrite(request: ChatRequest):

    rewrite_graph = build_rewrite_only_graph()

    result = await rewrite_graph.ainvoke({
        "query": request.query,
        "history": [item.model_dump() for item in request.history] if request.history else [],
        "awaiting_clarification": False,
        "clarification_context": None
    })

    return RewriteTestResponse(
        original_query=request.query,
        optimized_query=result.get("optimized_query")
           )

class GuardrailTestResponse(BaseModel):
    original_query: str
    out_of_scope: Optional[bool] = None
    blocked_response: Optional[str] = None    

@app.post("/test-guardrail", response_model=GuardrailTestResponse)
async def test_guardrail(request: ChatRequest):

    guardrail_graph = build_guardrail_graph()

    result = await guardrail_graph.ainvoke({
        "query": request.query,
        "history": [item.model_dump() for item in request.history] if request.history else [],
        "awaiting_clarification": False,
        "clarification_context": None
    })

    return GuardrailTestResponse(
        original_query=request.query,
        out_of_scope=result.get("out_of_scope"),
        blocked_response=result.get("final_response") if result.get("out_of_scope") else None
    )           

class RetrievedDocument(BaseModel):
    content: str
    score: Optional[float] = None
    source: Optional[str] = None
    department: Optional[str] = None


class RetrievalTestResponse(BaseModel):
    original_query: str
    optimized_query: Optional[str]
    departments: Optional[List[str]]
    out_of_scope: Optional[bool]
    retrieved_docs: Optional[List[RetrievedDocument]]

def build_department_finder_graph():
    graph = StateGraph(ShopState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("department", department_node)

    graph.set_entry_point("rewrite")

    graph.add_edge("rewrite", "guardrail")

    graph.add_conditional_edges(
        "guardrail",
        lambda state: "department" if not state.out_of_scope else END,
        {
            "department": "department",
            END: END
        }
    )

    graph.add_edge("department", END)

    return graph.compile()

@app.post("/test-department-picker", response_model=RetrievalTestResponse)
async def test_retrieval(request: ChatRequest):

    retrieval_graph = build_department_finder_graph()

    result = await retrieval_graph.ainvoke({
        "query": request.query,
        "history": [item.model_dump() for item in request.history] if request.history else [],
        "awaiting_clarification": False,
        "clarification_context": None
    })

    # Extract retrieved docs from state
    retrieved_docs = []

    if result.get("retrieved_docs"):
        for doc in result["retrieved_docs"]:
            retrieved_docs.append(
                RetrievedDocument(
                    content=doc.get("content"),
                    score=doc.get("score"),
                    source=doc.get("source"),
                    department=doc.get("department")
                )
            )

    return RetrievalTestResponse(
        original_query=request.query,
        optimized_query=result.get("optimized_query"),
        departments=result.get("departments"),
        out_of_scope=result.get("out_of_scope"),
        retrieved_docs=retrieved_docs
    )    

class DepartmentExecutionTestRequest(BaseModel):
    query: str
    departments: List[str]
    emotion: Optional[str] = "neutral"
    has_attachment: Optional[bool] = False
    attachment_text: Optional[str] = ""
    history: Optional[List[ConversationItem]] = []

class DepartmentExecutionTestResponse(BaseModel):
    query: str
    departments: Optional[List[str]]
    responses: Optional[List[str]]
    rag_docs_found: Optional[bool]
    turn_type: Optional[str]
    execution_node: Optional[str]

def build_department_execution_graph():
    graph = StateGraph(ShopState)

    graph.add_node("execute_departments", department_execution_node)

    graph.set_entry_point("execute_departments")
    graph.add_edge("execute_departments", END)

    return graph.compile()    

@app.post("/test-rag-retrieval", response_model=DepartmentExecutionTestResponse)
async def test_department_execution(request: DepartmentExecutionTestRequest):

    execution_graph = build_department_execution_graph()

    result = await execution_graph.ainvoke({
        "query": request.query,
        "optimized_query": request.query,
        "departments": request.departments,
        "emotion": request.emotion,
        "has_attachment": request.has_attachment,
        "attachment_text": request.attachment_text,
        "history": [item.model_dump() for item in request.history] if request.history else []
    })

    return DepartmentExecutionTestResponse(
        query=request.query,
        departments=request.departments,
        responses=result.get("responses"),
        rag_docs_found=result.get("rag_docs_found"),
        turn_type=result.get("turn_type"),
        execution_node=result.get("node_name")
    )    