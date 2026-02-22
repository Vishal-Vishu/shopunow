import json
import re
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

DEPARTMENTS = ["HR", "IT", "FACILITIES", "BILLING", "SHIPPING"]

BATCH_SIZE = 20
TOTAL_PER_DEPT = 100


PROMPT_TEMPLATE = """
Generate {batch_size} unique, enterprise-grade FAQ entries
for the {department} department of an e-commerce company called ShopUNow.

STRICT REQUIREMENTS:
- Each entry must contain:
    1. question (concise, realistic customer or employee query)
    2. answer (detailed, professional, 4–6 sentences minimum)
    3. topic (short label summarizing subject, 2–4 words)
    4. keywords (list of 3–6 important keywords)

- Questions must be realistic and varied.
- Answers must be policy-driven and operationally accurate.
- No duplicate or near-duplicate questions.
- Avoid generic phrasing.
- Cover edge cases, compliance, procedural and exception scenarios.
- Return strictly valid JSON.
- Do NOT include markdown.

Output format:
[
  {{
    "question": "...",
    "answer": "...",
    "category": "{department}",
    "topic": "...",
    "keywords": ["...", "...", "..."]
  }}
]
"""


def clean_json_response(text: str):
    """Attempt to clean and safely parse model JSON output."""
    try:
        return json.loads(text)
    except:
        # Remove common formatting artifacts
        text = re.sub(r"```json|```", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except:
            print("⚠ JSON parsing failed for batch.")
            return []


def generate_batch(department):
    prompt = PROMPT_TEMPLATE.format(
        department=department,
        batch_size=BATCH_SIZE
    )

    response = llm.invoke(prompt)
    return clean_json_response(response.content)


def main():
    full_dataset = []

    for dept in DEPARTMENTS:
        print(f"\nGenerating structured data for {dept}...")
        dept_records = []

        while len(dept_records) < TOTAL_PER_DEPT:
            batch = generate_batch(dept)
            dept_records.extend(batch)

            print(f"{dept}: {len(dept_records)} records generated so far...")

        # Trim to exact count
        dept_records = dept_records[:TOTAL_PER_DEPT]
        full_dataset.extend(dept_records)

    with open("shopunow_faq_dataset.json", "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, indent=4, ensure_ascii=False)

    print("\n✅ Structured dataset generated successfully!")
    print(f"Total records: {len(full_dataset)}")


if __name__ == "__main__":
    main()
