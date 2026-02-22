from langchain_openai import ChatOpenAI
from config import DEPARTMENTS
import json
import re
from graphbuilder import build_graph

graph = build_graph()

llm_router = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def safe_parse_json(response_text: str):
    """
    Cleans LLM markdown formatting and safely parses JSON.
    """

    # Remove markdown code fences if present
    cleaned = re.sub(r"```json|```", "", response_text).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def analyze_query(query: str, phone: str = "test_user", history=None):

    if history is None:
        history = []

    result = graph.invoke({
        "query": query,
        "phone": phone,
        "history": history,
        "user_sentiment": None,
        "departments": None,
        "responses": [],
        "final_response": None
    })

    return {
        "user_sentiment": result.get("sentiment"),
        "departments": result.get("departments"),
        "final_response": result.get("final_response")
    }