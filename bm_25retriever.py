import json
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, dataset_path):

        with open(dataset_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Partition by department
        self.department_data = {}
        self.department_bm25 = {}

        for item in self.data:
            dept = item.get("category")
            if not dept:
                continue

            self.department_data.setdefault(dept, []).append(item)

        # Build BM25 per department
        for dept, items in self.department_data.items():
            corpus = [
                (
                    item["question"] + " " +
                    " ".join(item.get("keywords", []))
                ).lower().split()
                for item in items
            ]
            self.department_bm25[dept] = BM25Okapi(corpus)

    def retrieve(self, query: str, department: str, top_k: int = 3):

        items = self.department_data.get(department)
        bm25 = self.department_bm25.get(department)

        if not items or not bm25:
            return []

        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        scored = list(zip(items, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [item for item, _ in scored[:top_k]]