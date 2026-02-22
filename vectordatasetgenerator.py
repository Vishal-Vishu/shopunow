import json
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import PINECONE_INDEX

load_dotenv()

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = PINECONE_INDEX
DATASET_PATH = "shopunow_faq_dataset.json"

EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 1536
METRIC = "cosine"

# ----------------------------------------------------
# Initialize Pinecone
# ----------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

def get_vector_store():

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes().names()

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # ------------------------------------------------
    # Case 1: Index DOES NOT exist → Create + Ingest
    # ------------------------------------------------
    if INDEX_NAME not in existing_indexes:

        print("Creating new Pinecone index...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Load dataset
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            faq_data = json.load(f)

        documents = []

        for item in faq_data:

            if not item.get("question") or not item.get("answer") or not item.get("category"):
                continue

            documents.append(
                Document(
                    page_content=item["question"],  # embed only question
                    metadata={
                        "category": item["category"],
                        "answer": item["answer"],
                        "topic": item.get("topic", ""),
                        "keywords": item.get("keywords", [])
                    }
                )
            )

        print(f"Ingesting {len(documents)} documents into new index...")

        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=INDEX_NAME
        )

        print("New index created and populated.")

    # ------------------------------------------------
    # Case 2: Index EXISTS → Just Connect
    # ------------------------------------------------
    else:
        print("Using existing Pinecone index.")

        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings
        )

    return vector_store

if __name__ == "__main__":
    get_vector_store()
