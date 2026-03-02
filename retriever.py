from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX
from opentelemetry import trace

# ==========================================================
# Configuration
# ==========================================================

SIMILARITY_THRESHOLD = 0.50      # Absolute threshold (optional safety)
TOP_K_RETRIEVE = 6               # Retrieve extra for ranking stability
TOP_K_USE = 3                    # Final docs passed to LLM

# ==========================================================
# Pinecone Setup
# ==========================================================

pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX,
    embedding=embeddings
)

tracer = trace.get_tracer(__name__)

# ==========================================================
# Hybrid Retrieval
# ==========================================================

def retrieve_docs(
    query: str,
    department: str,
    vector_store,
    bm25_retriever
) -> List[Document]:
    """
    Stable Hybrid Retrieval:
    - Pinecone vector similarity search (cosine)
    - Absolute threshold filtering
    - Deterministic top-k slicing
    - Optional BM25 merge
    - Deduplication
    - Stable ordering
    """

    with tracer.start_as_current_span("hybrid_retrieval") as span:

        span.set_attribute("query", query)
        span.set_attribute("department", department)

        # ==========================================================
        # 1️⃣ Vector Similarity Search (Stable)
        # ==========================================================

        with tracer.start_as_current_span("vector_search") as vspan:

            try:
                vector_results = vector_store.similarity_search_with_score(
                    query=query,
                    k=TOP_K_RETRIEVE,
                    filter={"category": department}
                )
            except Exception as e:
                vspan.record_exception(e)
                return []

            if not vector_results:
                vspan.set_attribute("vector_docs_count", 0)
                vector_docs = []
            else:
                # Sort descending by cosine similarity
                vector_results = sorted(
                    vector_results,
                    key=lambda x: x[1],
                    reverse=True
                )

                best_score = vector_results[0][1]
                vspan.set_attribute("best_vector_score", best_score)

                # Absolute threshold (prevents junk matches)
                # filtered = [
                #     (doc, score)
                #     for doc, score in vector_results
                #     if score >= SIMILARITY_THRESHOLD
                # ]

                # Deterministic top-k selection
                vector_docs = [
                    doc for doc, score in vector_results[:TOP_K_USE]
                ]

                vspan.set_attribute("vector_docs_count", len(vector_docs))

        print("=================================================")
        print("VECTOR DOCS")
        for d in vector_docs:
            print("-", d.metadata.get("topic", ""), "|", d.metadata.get("category", ""))
        print("=================================================")

        # ==========================================================
        # 2️⃣ BM25 Retrieval (Optional but Stable)
        # ==========================================================

        with tracer.start_as_current_span("bm25_search") as bspan:

            bm25_docs = []

            if bm25_retriever:
                try:
                    bm25_items = bm25_retriever.retrieve(
                        query=query,
                        department=department,
                        top_k=TOP_K_USE
                    )
                except Exception as e:
                    bspan.record_exception(e)
                    bm25_items = []

                for item in bm25_items:
                    try:
                        bm25_docs.append(
                            Document(
                                page_content=item["question"],
                                metadata={
                                    "category": item["category"],
                                    "answer": item["answer"],
                                    "topic": item.get("topic", ""),
                                    "keywords": item.get("keywords", [])
                                }
                            )
                        )
                    except KeyError:
                        continue

            print(f"[BM25] Retrieved {len(bm25_items)} raw items for dept={department}")        

            bspan.set_attribute("bm25_docs_count", len(bm25_docs))

        # ==========================================================
        # 3️⃣ Merge & Deduplicate (Stable Ordering)
        # ==========================================================

        combined = vector_docs + bm25_docs

        seen = set()
        final_docs = []

        for doc in combined:
            key = doc.page_content.strip()

            if key not in seen:
                seen.add(key)
                final_docs.append(doc)

        span.set_attribute("final_docs_count", len(final_docs))

        print("=================================================")
        print("FINAL DOCS USED")
        for d in final_docs[:TOP_K_USE]:
            print("-", d.metadata.get("topic", ""), "|", d.metadata.get("category", ""))
        print("=================================================")

        # Deterministic final slice
        return final_docs[:TOP_K_USE]