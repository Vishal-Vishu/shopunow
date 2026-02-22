import os
from typing import List, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.callbacks import BaseCallbackHandler
import json

import phoenix as px
from phoenix.otel import register
from phoenix.evals import create_classifier, evaluate_dataframe, LLM
from phoenix.experiments import run_experiment
from phoenix.client import AsyncClient

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "shopunow-structured-rag-index"

class DepartmentRouting(BaseModel):
    departments: List[
        Literal["HR", "IT", "FACILITIES", "BILLING", "SHIPPING"]
    ]

class GradeAnswer(BaseModel):
    """Score the quality of the generated answer."""
    faithfulness_score: int = Field(description="Score 1-5: 1 is hallucinated, 5 is fully grounded.")
    relevance_score: int = Field(description="Score 1-5: 1 is off-topic, 5 is perfectly relevant.")
    is_satisfactory: bool = Field(description="True if both scores are >= 4.")
    improvement_feedback: str = Field(description="Critique of what is missing or wrong.")

import logging
from langchain_core.callbacks import BaseCallbackHandler

# Setup standard Python logging
logging.basicConfig(level=logging.INFO, filename='shopunow_trace.log')
logger = logging.getLogger("ShopUNow")

class GraphBusinessLogger(BaseCallbackHandler):
    def on_chain_end(self, outputs, **kwargs):
        # Retrieve the marker we set in the node
        node_name = outputs.get("node_name", "unknown_node")
        
        print(f"✅ Node Trace: {node_name}")
        
        # Log specific data based on the node's responsibility
        if node_name == "department":
            print(f"   - Selected Depts: {outputs.get('departments')}")
        
        if node_name == "execute_departments":
            print(f"   - RAG Success: {outputs.get('rag_docs_found')}")
            
        if node_name == "answer_grader":
            print(f"   - Satisfactory: {outputs.get('is_satisfactory')}")
            print(f"   - Feedback: {outputs.get('improvement_feedback')}")
            
with open("taxonomy_registry.json", "r") as f:
    TAXONOMY = json.load(f)

CATEGORY_MAP = TAXONOMY["category_map"]

def setup_phoenix():
    """Configures Phoenix collector and API keys for tracing."""
    
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = ""#os.environ["PHOENIX_COLLECTOR_ENDPOINT"] 

    
    os.environ["PHOENIX_API_KEY"] = ""#os.environ["PHOENIX_API_KEY"] 

    
    os.environ["OPENAI_API_KEY"] = ""#os.environ["OPENAI_API_KEY"] 

    # Register the tracer for the project
    tracer_provider = register(
        project_name="shopunow",
        auto_instrument=True,
    )
    
    return tracer_provider
