from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from graphbuilder import build_graph
from multimodalprocessor import process_uploaded_file

# accessing through uvicorn -- uvicorn api:app --host 127.0.0.1 --port 8000


# ==========================================================
# Initialize FastAPI
# ==========================================================

app = FastAPI(
    title="ShopUNow Agentic AI API",
    version="1.0.0"
)

graph = build_graph()


# ==========================================================
# Request Schema
# ==========================================================

class ChatRequest(BaseModel):
    query: str
    phone: Optional[str] = ""
    history: Optional[List[dict]] = []


class ChatResponse(BaseModel):
    response: str


# ==========================================================
# Text-Only Endpoint
# ==========================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    print("Chat from FASt api")
    result = graph.invoke({
        "query": request.query,
        "phone": request.phone,
        "history": request.history,
        "sentiment": None,
        "departments": None,
        "responses": None,
        "final_response": None
    })

    return ChatResponse(response=result.get("final_response", ""))


# ==========================================================
# Multimodal Endpoint
# ==========================================================

@app.post("/chat-with-file", response_model=ChatResponse)
async def chat_with_file(
    query: str = Form(...),
    phone: str = Form(""),
    file: UploadFile = File(None)
):

    combined_query = query

    if file:
        extracted_text = process_uploaded_file(file.file)

        combined_query = f"""
User Query:
{query}

Attached Document Content:
{extracted_text}
"""

    result = graph.invoke({
        "query": combined_query,
        "phone": phone,
        "history": [],
        "sentiment": None,
        "departments": None,
        "responses": None,
        "final_response": None
    })

    return ChatResponse(response=result.get("final_response", ""))


# ==========================================================
# Health Check
# ==========================================================

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
