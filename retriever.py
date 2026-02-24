from typing import List
from langchain_core.documents import Document
from pinecone import Pinecone
from opentelemetry import trace
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


SIMILARITY_THRESHOLD = 0.5
TOP_K_RETRIEVE = 4
TOP_K_USE = 3

tracer = trace.get_tracer(__name__)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def extract_keywords(query: str):
    # Simple fast tokenizer (can improve later)
    return [token.strip().lower() for token in query.split() if len(token) > 2]


def retrieve_docs(
    query: str,
    department: str,
    vector_store,
    bm25_retriever
) -> List[Document]:

    with tracer.start_as_current_span("hybrid_retrieval") as span:

        span.set_attribute("query", query)
        span.set_attribute("department", department)

        query_embedding = embeddings.embed_query(query)

        index = vector_store._index  # reuse underlying Pinecone index

        query_tokens = extract_keywords(query)

        # -----------------------------------------------------
        # 1️⃣ Vector Search with Keyword Filter
        # -----------------------------------------------------

        with tracer.start_as_current_span("vector_search") as vspan:

            metadata_filter = {
                "category": department
            }

            # Add keyword filter if tokens exist
            if query_tokens:
                metadata_filter["keywords"] = {"$in": query_tokens}

            try:
                vector_response = index.query(
                    vector=query_embedding,
                    top_k=TOP_K_RETRIEVE,
                    include_metadata=True,
                    include_values=False,
                    filter=metadata_filter
                )
            except Exception as e:
                vspan.record_exception(e)
                vector_response = None

            matches = vector_response.get("matches", []) if vector_response else []

            # Fallback to category-only if no keyword matches
            if not matches:
                try:
                    vector_response = index.query(
                        vector=query_embedding,
                        top_k=TOP_K_RETRIEVE,
                        include_metadata=True,
                        include_values=False,
                        filter={"category": department}
                    )
                    matches = vector_response.get("matches", [])
                except Exception as e:
                    vspan.record_exception(e)
                    matches = []

            vector_docs = []

            if matches:
                best_score = matches[0]["score"]
                vspan.set_attribute("best_vector_score", best_score)

                if best_score >= SIMILARITY_THRESHOLD:
                    for match in matches:
                        if match["score"] >= best_score * 0.95:
                            metadata = match["metadata"]

                            vector_docs.append(
                                Document(
                                    page_content=metadata.get("text", ""),
                                    metadata=metadata
                                )
                            )

            vspan.set_attribute("vector_docs_count", len(vector_docs))

        # -----------------------------------------------------
        # 2️⃣ Optimized BM25 Retrieval
        # -----------------------------------------------------

        with tracer.start_as_current_span("bm25_search") as bspan:

            try:
                bm25_items = bm25_retriever.retrieve(
                    query=query,
                    department=department,
                    top_k=TOP_K_USE
                )
            except Exception as e:
                bspan.record_exception(e)
                bm25_items = []

            bm25_docs = [
                Document(
                    page_content=item["question"],
                    metadata={
                        "answer": item.get("answer", ""),
                        "category": item.get("category", ""),
                        "topic": item.get("topic", ""),
                        "keywords": item.get("keywords", [])
                    }
                )
                for item in bm25_items
            ]

            bspan.set_attribute("bm25_docs_count", len(bm25_docs))

        # -----------------------------------------------------
        # 3️⃣ Merge & Deduplicate
        # -----------------------------------------------------

        combined = vector_docs + bm25_docs

        seen = set()
        final_docs = []

        for doc in combined:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                final_docs.append(doc)

        span.set_attribute("final_docs_count", len(final_docs))

        return final_docs[:TOP_K_USE]