
import json
from rank_bm25 import BM25Okapi

class BM25Retriever:

    def __init__(self, dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Tokenize questions
        self.corpus = [item["question"].lower().split() for item in self.data]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, department:str , top_k: int=3):
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        scored_items = list(zip(self.data, scores))

        # Filter by department
        filtered = [
            (item, score)
            for item, score in scored_items
            if item["category"] == department
        ]

        # Sort by BM25 score
        filtered.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        return [item for item, _ in filtered[:top_k]]
