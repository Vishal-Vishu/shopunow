
# 🛍️ ShopUNow – Agentic AI Customer Support System

ShopUNow is an advanced **LangGraph-powered multi-department AI support system** built using:

- LangGraph (Agent orchestration)
- LangChain
- OpenAI (GPT-4o)
- Pinecone (Vector DB)
- BM25 (Keyword Retrieval)
- FastAPI (API backend)
- Streamlit (UI)
- SQLite (Session & Support storage)
- Multimodal Processing (PDF / DOCX / Images via Vision LLM)

---

## 🚀 Overview

ShopUNow is a multi-department intelligent RAG system that:

- Classifies queries into departments (HR, IT, Billing, Shipping, Facilities)
- Applies guardrails (gibberish / out-of-scope detection)
- Performs hybrid retrieval (BM25 + Pinecone)
- Generates department-specific responses
- Performs faithfulness & relevance scoring
- Detects sentiment & emotion
- Supports clarification loops
- Stores persistent session history
- Handles multimodal uploads (PDF, DOCX, images)
- Supports session-based conversation tracking

---

## 🧠 Architecture Flow

User Query  
→ Guardrail Node  
→ Rewrite Node  
→ Department Picker  
→ Parallel RAG Execution  
→ Faithfulness Check  
→ Final Response Composer  
→ Memory + Sentiment Logging  

---

## 📂 Project Structure

ShopUNow/
- app.py
- api.py
- graphbuilder.py
- graphnodes.py
- graphstate.py
- llmagents.py
- bm_25retriever.py
- ragvectorstoretester.py
- billing_db_tool.py
- memory.py
- multimodalprocessor.py
- taxonomy_registry.json
- support_tickets.db
- requirements.txt
- ShopUNow.postman_collection.json

---

## 🧩 Core Components

### Graph State
Central state model using Pydantic with custom reducer to prevent merge conflicts.

### Hybrid Retrieval
- BM25 keyword search
- Pinecone vector semantic search
- Embedding model: text-embedding-3-small
- LLM: gpt-4o

### Multimodal Processing
- PDF extraction
- DOCX extraction
- OCR
- GPT-4o Vision for semantic image understanding

### Billing Database Tool
- SQLite-based order lookup
- Extract order IDs
- Fetch order details and items

### Conversation Memory System
- Session ID tracking
- Phone-based tracking
- Sentiment logging
- Persistent conversation history

### Department Taxonomy
- Keyword → Category → Topic mapping
- Multi-department routing
- Topic shift detection

---

## ⚙️ Installation

1. Clone repository
2. Create virtual environment
3. Install dependencies

pip install -r requirements.txt

---

## 🔑 Environment Variables

Create .env file:

OPENAI_API_KEY=your_key  
PINECONE_API_KEY=your_key  

---

## ▶️ Running the System

Start FastAPI:

uvicorn api:app --reload

Start Streamlit:

streamlit run app.py

---

## 🧪 Testing

- Use Postman collection
- Run ragvectorstoretester.py for Pinecone testing

---

## 🧠 Guardrail Capabilities

- Gibberish detection
- Out-of-scope filtering
- Topic shift detection
- Escalation logic
- Clarification loops

---

## 💾 Database Schema

conversation_messages:
- id
- phone
- session_id
- query
- response
- turn_type
- department
- sentiment
- emotion
- created_at

---

## 📊 Observability

Supports:
- LangSmith
- Arize Phoenix
- OpenInference
- OpenTelemetry

---

## 🌍 Deployment

- Render deployment ready
- Streamlit cloud compatible
- Configure environment variables
- Ensure SQLite persistence strategy

---

## 🛠️ Future Enhancements

- Redis session backend
- Async department execution
- Response ranking
- Tool calling integration
- Vector cache layer
- Rate limiting
- Structured JSON responses

---

## 👨‍💻 Author

Vishal V  
Agentic AI Systems – LangGraph + RAG Architecture
