from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX

from opentelemetry import trace

pc = Pinecone(api_key=PINECONE_API_KEY)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX,
    embedding=embeddings
)

SIMILARITY_THRESHOLD = 0.5
TOP_K_RETRIEVE = 6
TOP_K_USE = 3

#bm25_retriever = BM25Retriever("shopunow_faq_dataset.json")



tracer = trace.get_tracer(__name__)


def retrieve_docs(
    query: str,
    department: str,
    vector_store,
    bm25_retriever
) -> List[Document]:
    """
    Hybrid RAG Retrieval:
    - Pinecone vector similarity search (cosine)
    - BM25 keyword retrieval
    - Dynamic score filtering
    - Deduplication
    - Phoenix tracing enabled
    """

    with tracer.start_as_current_span("hybrid_retrieval") as span:

        span.set_attribute("query", query)
        span.set_attribute("department", department)

       
        with tracer.start_as_current_span("vector_search") as vspan:

            try:
                vector_results = vector_store.similarity_search_with_score(
                    query,
                    k=TOP_K_RETRIEVE,
                    filter={"category": department},
                    
                )
            except Exception as e:
                vspan.record_exception(e)
                return []

            if not vector_results:
                vspan.set_attribute("vector_docs_count", 0)
                vector_docs = []
                best_score = 0
            else:
                # Sort descending (cosine: higher = better)
                vector_results = sorted(vector_results, key=lambda x: x[1], reverse=True)

                best_score = vector_results[0][1]
                vspan.set_attribute("best_vector_score", best_score)

                # Hard reject weak matches
                if best_score < SIMILARITY_THRESHOLD:
                    vector_docs = []
                else:
                    # Relative filtering (within 95% of best score)
                    vector_docs = [
                        doc
                        for doc, score in vector_results
                        if score >= best_score * 0.95
                    ]

                vspan.set_attribute("vector_docs_count", len(vector_docs))
        print("*******************************************************")                
        print("Vector Docs")
        print(vector_docs)
        print("*******************************************************")                

        # -----------------------------------------------------
        # 2️⃣ BM25 Retrieval
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

            bm25_docs = []

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

            bspan.set_attribute("bm25_docs_count", len(bm25_docs))

        # -----------------------------------------------------
        # 3️⃣ Merge & Deduplicate
        # -----------------------------------------------------
        combined = vector_docs + bm25_docs

        seen_questions = set()
        final_docs = []

        for doc in combined:
            question_text = doc.page_content.strip()
            if question_text not in seen_questions:
                final_docs.append(doc)
                seen_questions.add(question_text)

        print("******************************************************************")
        print("Final Docs")
        print("******************************************************************")

        span.set_attribute("final_docs_count", len(final_docs))
        

        return final_docs[:TOP_K_USE]


def test_rag(query, department):

    print("\n==============================")
    print(f"Query: {query}")
    print(f"Department: {department}")
    print("==============================\n")

    docs = retrieve_docs(
        query=query,
        department=department,
        vector_store=vector_store,
        bm25_retriever=bm25_retriever
    )

    if not docs:
        print("❌ No documents retrieved.\n")
        return

    print(f"✅ Retrieved {len(docs)} documents:\n")

    for i, doc in enumerate(docs, 1):
        print(f"--- Document {i} ---")
        print("Question:", doc.page_content)
        print("Answer:", doc.metadata.get("answer", ""))
        print()


# -----------------------------------
# Run Tests
# -----------------------------------
if __name__ == "__main__":

    test_rag("request a leave of absence", "HR")

    test_rag("How can I request a leave of absence for personal reasons", "HR")

    test_rag("authorization hold", "BILLING")

    test_rag("refund", "BILLING") 