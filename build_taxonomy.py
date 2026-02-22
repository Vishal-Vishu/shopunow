import json
import re
from collections import defaultdict
from pathlib import Path


DATASET_PATH = "shopunow_faq_dataset.json"
OUTPUT_PATH = "taxonomy_registry.json"


def normalize(text: str):
    return re.sub(r"\s+", " ", text.strip().lower())


def build_registry():

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_keywords = set()
    category_map = defaultdict(set)
    topic_map = defaultdict(set)

    for item in data:

        category = normalize(item.get("category", "unknown"))
        topic = normalize(item.get("topic", "unknown"))

        keywords = item.get("keywords", [])

        for kw in keywords:
            kw_norm = normalize(kw)

            all_keywords.add(kw_norm)
            category_map[category].add(kw_norm)
            topic_map[topic].add(kw_norm)

    registry = {
        "all_keywords": sorted(list(all_keywords)),
        "category_map": {k: sorted(list(v)) for k, v in category_map.items()},
        "topic_map": {k: sorted(list(v)) for k, v in topic_map.items()}
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4)

    print(f"✅ Taxonomy registry created at {OUTPUT_PATH}")
    print(f"Total unique keywords: {len(all_keywords)}")


if __name__ == "__main__":
    build_registry()
