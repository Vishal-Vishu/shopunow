import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# 1. Setup Connection
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


index_name = "shopunow-rag-index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Connect to the Existing Vector Store
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
)

def test_rag_query(query, department):
    print(f"\n--- Testing Query [{department}]: {query} ---")
    
    # Define the filter for the specific department
    search_filter = {"category": department}
    
    # 1. Test Retrieval (The 'R' in RAG)
    # This checks if Pinecone finds the right records
    retriever = vectorstore.as_retriever(search_kwargs={'filter': search_filter, 'k': 2})
    source_docs = retriever.invoke(query)
    
    print(f"Retrieved {len(source_docs)} documents from Pinecone.")
    for i, doc in enumerate(source_docs):
        print(f"Source {i+1} (Category: {doc.metadata['category']}): {doc.page_content[:100]}...")

    # 2. Test Generation (The 'G' in RAG)
    # This checks if the LLM uses that context to answer
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    response = qa_chain.invoke(query)
    print(f"FINAL ANSWER: {response['result']}")

# --- RUN TESTS ---

# Test HR Category
test_rag_query("How do I apply for PTO?", "HR")

# Test IT Category
test_rag_query("How do I reset my password?", "IT")

# Test Billing Category
test_rag_query("What payment methods do you accept?", "Billing")