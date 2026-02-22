from bm_25retriever import BM25Retriever

bm25_retriever = BM25Retriever("shopunow_faq_dataset.json")

print(type(bm25_retriever))  # Should show BM25Retriever

result = bm25_retriever.retrieve(
    query="leave policy",
    department="HR"
)

print(result)
