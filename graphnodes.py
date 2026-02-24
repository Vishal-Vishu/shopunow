from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from retriever import retrieve_docs
from llmagents import AGENT_MAP
from graphstate import ShopState
from vectordatasetgenerator import get_vector_store
import json
from pydantic import BaseModel
from bm_25retriever import BM25Retriever
import re

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from config import DepartmentRouting, GradeAnswer, CATEGORY_MAP
from opentelemetry import trace

from graphstate import AffectiveOutput



bm25_retriever = BM25Retriever("shopunow_faq_dataset.json")

tracer = trace.get_tracer(__name__)

with open("taxonomy_registry.json", "r", encoding="utf-8") as f:
    TAXONOMY = json.load(f)

KNOWN_KEYWORDS = set(TAXONOMY["all_keywords"])


def normalize(text: str):
    return re.sub(r"\s+", " ", text.strip().lower())

class GuardrailOutput(BaseModel):
    out_of_scope: bool
    reason: str

class RewriteOutput(BaseModel):
    needs_rewrite: bool
    rewritten_query: str    

def guardrail_node(state: ShopState):

    print("Guard Rail node started")

    effective_query = state.optimized_query or state.query

    query = normalize(effective_query)

    # -----------------------------------------
    # 1️⃣ Rule-Based Injection Detection
    # -----------------------------------------

    injection_patterns = [
        "ignore previous instructions",
        "act as system",
        "bypass",
        "jailbreak",
        "developer mode",
        "hack",
        "exploit",
        "reveal system prompt",
        "override policy"
    ]

    for pattern in injection_patterns:
        if pattern in query:
            return {
                "out_of_scope": True,
                "guardrail_reason": "Prompt injection or malicious intent detected."
            }
        
    for keyword in KNOWN_KEYWORDS:
        if keyword in query:
            return {
                "out_of_scope": False,
                "guardrail_reason": "Matched known taxonomy keyword.",
                "node_name": "Guard Rail Node"
            }    

    # -----------------------------------------
    # 2️⃣ LLM-Based Scope Classification
    # -----------------------------------------

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    ).with_structured_output(GuardrailOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a domain guardrail classifier for ShopUNow AI assistant.

ShopUNow supports:
- Orders
- Refunds
- Returns
- Delivery issues
- Product information
- Account issues
- Billing
- Facility services (if applicable)

If the query is unrelated to these business domains,
mark it as out_of_scope = true.

Examples of out_of_scope:
- Legal advice
- Medical advice
- Political topics
- Coding help
- General knowledge
- Hacking
- Religion
- Personal relationship advice
         
IMPORTANT:
- Customer complaints or criticism about products/services are IN-SCOPE.
- Negative feedback is NOT out_of_scope.
- Emotional language alone does NOT make it out_of_scope.
- If the current query is vague, incomplete, or depends on previous conversation,
you MUST expand it into a fully self-contained question using conversation context.

Example:

History:
User: refund details
Current: where can I check
→ "Where can I check the status of my refund?"         

Only mark out_of_scope = true if the topic is completely unrelated
to ShopUNow business operations.

When unsure mark out_of_scope = false                  

Return structured output only.
"""),
        ("human", "{query}")
    ])

    chain = prompt | llm
    result = chain.invoke({"query": state.query})

    print("Guard Rail Reason =",result.reason)
    print("Guard Rail node finished")

    return {
        "out_of_scope": result.out_of_scope,
        "guardrail_reason": result.reason,
        "node_name": "rewrite node"
    }    

def rewrite_node(state: ShopState):

    with tracer.start_as_current_span("rewrite_node"):

        span = trace.get_current_span()

        # ==================================================
        # 1️⃣ Clarification Handling (INTERCEPT EARLY)
        # ==================================================

        if state.awaiting_clarification:

            print("Handling clarification response")

            user_reply = state.query.lower().strip()

            clarification_ctx = state.clarification_context or {}
            previous_query = clarification_ctx.get("previous_query")
            new_query = clarification_ctx.get("new_query")

            # ---- Confirm Topic Shift ----
            if any(word in user_reply for word in ["yes", "yeah", "correct", "right"]):

                print("User confirmed topic shift")

                return {
                    # 🔥 CRITICAL: Override query itself
                    "query": new_query,
                    "optimized_query": new_query,
                    "awaiting_clarification": False,
                    "clarification_context": None,
                    "needs_rewrite": False,
                    "node_name": "rewrite node"
                }

            # ---- Reject Topic Shift ----
            if any(word in user_reply for word in ["no", "not"]):

                print("User rejected topic shift")

                return {
                    # 🔥 CRITICAL: Revert to previous query
                    "query": previous_query,
                    "optimized_query": previous_query,
                    "awaiting_clarification": False,
                    "clarification_context": None,
                    "needs_rewrite": False,
                    "node_name": "rewrite node"
                }

            # ---- Unclear Confirmation ----
            print("Unclear clarification response")

            return {
                "final_response": "Could you please confirm whether you are switching to the new issue?",
                "awaiting_clarification": True,
                "node_name": "rewrite node"
            }

        # ==================================================
        # 2️⃣ Normal Rewrite Flow
        # ==================================================

        print("Rewrite Node")

        query = state.query.strip()
        span.set_attribute("query_used", query)

        # --------------------------------------------------
        # Build conversation context
        # --------------------------------------------------

        conversation_context = build_conversation_context(state)
        has_context = bool(conversation_context.strip())

        # --------------------------------------------------
        # Cheap Skip Rule
        # --------------------------------------------------

        if len(query.split()) > 6 and query.endswith("?") and not has_context:

            return {
                "needs_rewrite": False,
                "optimized_query": query,
                "node_name": "rewrite node"
            }

        # --------------------------------------------------
        # Context-Aware LLM Rewrite
        # --------------------------------------------------

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        ).with_structured_output(RewriteOutput)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a context-aware query rewriting assistant for ShopUNow.

Your task:
- Rewrite the CURRENT USER QUERY into a clear, fully-formed question.
- Use conversation history to resolve ambiguity.
- Preserve original emotional tone.
- Do NOT change user intent.
- Do NOT neutralize complaints.
- Do NOT invent new information.

If the query is already clear and self-contained,
set needs_rewrite = false.

Return structured output only.
"""),
            ("human", """
Conversation Context:
{context}

Current User Query:
{query}
""")
        ])

        chain = prompt | llm

        result = chain.invoke({
            "context": conversation_context,
            "query": query
        })

        span.set_attribute("needs_rewrite", result.needs_rewrite)
        span.set_attribute("optimized_query", result.rewritten_query)

        print("Rewrite Result:", result)

        return {
            "needs_rewrite": result.needs_rewrite,
            "optimized_query": result.rewritten_query if result.needs_rewrite else query,
            "node_name": "rewrite node"
        }
        
def preprocess_node(state):
    print("Preprocess node begins execution")
    conversation_context = ""

    for item in state.history:
        conversation_context += f"\nUser: {item.query}"
        conversation_context += f"\nAssistant: {item.response}"

    conversation_context += f"\nUser: {state.query}"

    print("Preprocess node finishes execution= ",conversation_context)

    return {"context": conversation_context, "node_name": "preprocess"}

def validate_input_node(state):

    # If already validated model, just return it
    if isinstance(state, ShopState):
        return state.model_dump()

    # If dict, validate and normalize
    validated = ShopState(**state)

    return validated.model_dump()


def build_conversation_context(state: ShopState, max_turns: int = 2):

    print("Build conversation context")

    history = state.history

    if not history:
        return ""

    # --------------------------------------------------
    # 1️⃣ Separate long-term & short-term memory
    # --------------------------------------------------
    valid_history = [
        item for item in history 
        if getattr(item, "turn_type", "success") != "system_failure"
    ]

    # Recent Detailed Turns (using only valid items)
    
    long_term = history[:-max_turns] if len(history) > max_turns else []
    recent_history = valid_history[-max_turns:]

    context_parts = []

    # --------------------------------------------------
    # 2️⃣ Compressed Long-Term Memory
    # --------------------------------------------------

    if long_term:
        summary_lines = []
        for item in long_term:
            summary_lines.append(f"User previously asked about: {item.query}")

        compressed_summary = "\n".join(summary_lines[-5:])  # cap summary size

        context_parts.append("Conversation Summary:")
        context_parts.append(compressed_summary)

    # --------------------------------------------------
    # 3️⃣ Recent Detailed Turns
    # --------------------------------------------------

    context_parts.append("\nRecent Conversation:")

    for item in recent_history:
        context_parts.append(f"User: {item.query}")
        context_parts.append(f"Assistant: {item.response}")

    # --------------------------------------------------
    # 4️⃣ Include Optimized Query If Exists
    # --------------------------------------------------

    if getattr(state, "optimized_query", None):
        context_parts.append(
            f"\nClarified Current Query: {state.optimized_query}"
        )

    final_context = "\n".join(context_parts)

    print("Conversation context finished=", final_context)

    return final_context.strip()
            

def sentiment_node(state: ShopState):

    print("Affective Analysis Node Activated")

    conversation_context = build_conversation_context(state)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    ).with_structured_output(AffectiveOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an affective analysis engine for ShopUNow.

Analyze the user's current emotional state.

Return:

1. sentiment: positive | neutral | negative
2. emotion: choose exactly ONE from:
   [anger, frustration, confusion, fear, sadness, disappointment,
    neutral, satisfaction, gratitude]
3. emotion_intensity: float between 0 and 1
4. requires_escalation: true if:
   - strong anger
   - repeated complaint tone
   - legal threats
   - emotion_intensity > 0.75 and emotion in [anger, frustration]

Rules:
- Informational queries are neutral.
- Asking for help calmly is neutral.
- Do NOT over-classify as negative.
- Be conservative and precise.

Return structured output only.
"""),
        ("human", "{context}")
    ])

    chain = prompt | llm

    try:
        result = chain.invoke({"context": conversation_context})

        print("Affective Result:", result)

        return {
            "sentiment": result.sentiment,
            "emotion": result.emotion,
            "emotion_intensity": result.emotion_intensity,
            "escalation_required": result.requires_escalation,
            "node_name": "affective node"
        }

    except Exception as e:
        print("Affective analysis error:", e)

        # Safe fallback
        return {
            "sentiment": "neutral",
            "emotion": "neutral",
            "emotion_intensity": 0.0,
            "escalation_required": False,
            "node_name": "affective node"
        }


def sentiment_node_old(state: ShopState):

    print("Detecting sentiment")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    conversation_context = build_conversation_context(state)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a sentiment classification and escalation router for ShopUNow.
Analyze the user's message to determine if they are currently upset or just asking a question.

DEFINITIONS:
- Positive: Gratitude, praise, or excitement (e.g., "Thank you!", "Great service").
- Neutral: Informational queries, "How-to" questions, or factual statements (e.g., "Where is my order?", "How do I return this?").
- Negative: Expressions of anger, frustration, dissatisfaction, or threats (e.g., "This is broken!", "I've been waiting for hours!").

EXAMPLES:
- "How do I track my order?" -> neutral
- "My payment failed, please help." -> neutral (seeking help, not yet angry)
- "I am so sick of these delays!" -> negative
- "You guys are the best!" -> positive

Return ONLY valid JSON: {{"sentiment": "positive | neutral | negative"}}
         
Do not include extra text.
"""),
        ("human", "{context}")
    ])

    chain = prompt | llm

    response = chain.invoke({"context": conversation_context}).content
    
    print("Sentiment Node Response : ", response)

    try:
        parsed = json.loads(response)
        sentiment = parsed.get("sentiment", "negative")
        
        # Safety fallback
        if sentiment not in ["positive", "neutral", "negative"]:
            sentiment = "negative"
            escalation_required = True

    except Exception:
        # If model output breaks → escalate
        sentiment = "neutral"

    escalation_required = (sentiment == "negative")

    print("Detected sentiment =", sentiment)

    return {
        "sentiment": sentiment,
        "escalation_required": escalation_required,
        "node_name": "sentiment node"
    }




# ---------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------






def build_routing_context(state: ShopState):

    current_query = state.optimized_query or state.query
    history = state.history or []

    # If no previous turns, route on current query only
    if not history:
        return f"Current User Query:\n{current_query}"

    # Only include last USER query (not assistant response)
    last_turn = history[-1]

    return f"""
Previous User Query:
{last_turn.query}

Current User Query:
{current_query}
""".strip()



import json
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Literal


# ---------------------------------------------------------
# Load taxonomy once
# ---------------------------------------------------------

with open("taxonomy_registry.json", "r") as f:
    TAXONOMY = json.load(f)

CATEGORY_MAP = TAXONOMY["category_map"]


# ---------------------------------------------------------
# Structured Output for LLM
# ---------------------------------------------------------

class DepartmentRouting(BaseModel):
    departments: List[
        Literal["HR", "IT", "FACILITIES", "BILLING", "SHIPPING"]
    ]


# ---------------------------------------------------------
# Taxonomy Scoring
# ---------------------------------------------------------

def taxonomy_department_scoring(query: str):

    query_lower = query.lower()
    scores = defaultdict(int)

    for dept, keywords in CATEGORY_MAP.items():

        if dept == "unknown":
            continue

        for keyword in keywords:
            if keyword.lower() in query_lower:
                scores[dept.upper()] += 1

    return scores


from collections import defaultdict

def detect_department_from_taxonomy(query: str):

    query_lower = query.lower()
    scores = defaultdict(int)

    for dept, keywords in CATEGORY_MAP.items():

        if dept == "unknown":
            continue

        for keyword in keywords:
            if keyword.lower() in query_lower:
                scores[dept.upper()] += 1

    if not scores:
        return []

    max_score = max(scores.values())

    # return departments within 80% of max score
    return [
        dept for dept, score in scores.items()
        if score >= max_score * 0.8
    ]

def department_node(state: ShopState):

    print("Hybrid Department Routing Node Executed")

    query = build_routing_context(state)
    print("Routing Query:", query)

    # --------------------------------------------------
    # 1️⃣ Taxonomy Scoring
    # --------------------------------------------------

    taxonomy_scores = taxonomy_department_scoring(query)

    print("Taxonomy Scores:", dict(taxonomy_scores))

    # Normalize taxonomy selection
    taxonomy_departments = []

    if taxonomy_scores:
        max_score = max(taxonomy_scores.values())

        taxonomy_departments = [
            dept
            for dept, score in taxonomy_scores.items()
            if score >= max_score * 0.8  # relative threshold
        ]

    print("Taxonomy Departments:", taxonomy_departments)

    # --------------------------------------------------
    # 2️⃣ LLM Semantic Routing
    # --------------------------------------------------

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    ).with_structured_output(DepartmentRouting)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a strict routing classifier for ShopUNow.

Determine which department(s) should handle the query.

Valid departments:
HR, IT, FACILITIES, BILLING, SHIPPING

Rules:
- Base decision only on the current query.
- Do not guess.
- If unclear, return [].
- Return structured output only.
"""),
        ("human", "{query}")
    ])

    chain = prompt | llm

    try:
        llm_result = chain.invoke({"query": query})
        llm_departments = llm_result.departments
    except Exception as e:
        print("LLM routing error:", e)
        llm_departments = []

    print("LLM Departments:", llm_departments)

    # --------------------------------------------------
    # 3️⃣ Hybrid Merge Logic
    # --------------------------------------------------

    final_departments = set()

    # Case A: Both agree
    intersection = set(taxonomy_departments) & set(llm_departments)

    if intersection:
        final_departments = intersection

    # Case B: Taxonomy strong but LLM empty
    elif taxonomy_departments and not llm_departments:
        final_departments = set(taxonomy_departments)

    # Case C: LLM predicts but taxonomy weak
    elif llm_departments and not taxonomy_departments:
        final_departments = set(llm_departments)

    # Case D: Both predict but different
    elif taxonomy_departments and llm_departments:
        final_departments = set(taxonomy_departments) | set(llm_departments)

    else:
        final_departments = set()

    final_departments = list(final_departments)

    print("Final Hybrid Departments:", final_departments)

    return {
        "departments": final_departments,
        "node_name": "department node"
    }

def escalation_node(state: ShopState):

    llm = ChatOpenAI(model="gpt-4o-mini")

    conversation_context = build_conversation_context(state)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a customer support escalation assistant.
        Apologize professionally and inform the user
        that a human agent will contact them shortly.
        """),
        ("human", "{context}")
    ])

    chain = prompt | llm
    response = chain.invoke({"context": conversation_context}).content

    return {"final_response": response}




async def execute_single_department(dept, latest_query, context, vectorstore, bm25_retriever, state):
    """
    Helper coroutine to handle RAG and LLM generation for one specific department.
    """
    # 1. Retrieval (If retrieve_docs is synchronous, it's fine to call here, 
    # but the LLM call below will be awaited asynchronously)
    docs = retrieve_docs(
        query=latest_query,
        department=dept,
        vector_store=vectorstore,
        bm25_retriever=bm25_retriever
    )

    if not docs:
        print(f"[System Failure] No RAG docs for {dept}")
        return None

    # Use ANSWERS from metadata
    rag_context = "\n\n".join([doc.metadata.get("answer", "") for doc in docs])

    if not rag_context.strip():
        return None

    llm, prompt = AGENT_MAP[dept]

    tone_instruction = ""

    if state.emotion in ["anger", "frustration"]:
        tone_instruction = "Respond in a calm, empathetic and apologetic tone."
    elif state.emotion == "confusion":
        tone_instruction = "Respond clearly with step-by-step explanation."
    elif state.emotion == "fear":
        tone_instruction = "Respond in a reassuring and confident tone."


    enhanced_query = f"""
PRIMARY TASK: Answer the following user query: {latest_query}
---
KNOWLEDGE BASE: {rag_context}
CONTEXT INFO: {context}
---
INSTRUCTIONS:
- Base answer strictly on knowledge base using {dept} persona.
- If no info exists, say: "This information is not available in our records."
- {tone_instruction}
"""

    # 2. Async LLM Call: using .ainvoke instead of .invoke
    chain = prompt | llm
    result = await chain.ainvoke({"query": enhanced_query})

    if "not available in our records" in result.content:
        print(f"⚠️ [System Failure] LLM confirmed no info for {dept}")
        return None
        
    return result.content

async def department_execution_node(state: ShopState):
    print(f"Department Execution started with query == {state.query}")

    context = build_conversation_context(state)
    vectorstore = get_vector_store()
    latest_query = state.optimized_query or state.query

    # 3. Create tasks for all identified departments
    tasks = [
        execute_single_department(dept, latest_query, context, vectorstore, bm25_retriever, state)
        for dept in state.departments
    ]

    # 4. Execute all departments concurrently
    results = await asyncio.gather(*tasks)

    # Filter out None values from failed retrievals or records not found
    responses = [res for res in results if res is not None]

    if not responses:
        return {
            "responses": ["I'm sorry, I couldn't find any specific information regarding that in our ShopUNow records."],
            "rag_docs_found": False,
            "turn_type": "knowledge_gap",
            "node_name": "Department Execution"
        }

    return {
        "responses": responses,
        "rag_docs_found": True,
        "turn_type": "success",
        "node_name": "Department Execution"
    }

def department_execution_node1(state: ShopState):

    print("Department Execution started with query == ", {state.query})

    context = build_conversation_context(state)
    vectorstore = get_vector_store()

    responses = []
    
    latest_query = state.optimized_query or state.query

    for dept in state.departments:

        docs = retrieve_docs(
            query=latest_query,
            department=dept,
            vector_store=vectorstore,
            bm25_retriever=bm25_retriever
        )

        #docs = retrieve_docs(latest_query, dept)

        print(f"[RAG] Department: {dept} | Docs Retrieved: {len(docs)}")

        if not docs:
            print(f"[System Failure] No RAG docs for {dept}")
            continue

        any_docs_found = True

        # Use ANSWERS, not questions
        rag_context = "\n\n".join(
            [doc.metadata.get("answer", "") for doc in docs]
        )

        if not rag_context.strip():
            continue

        llm, prompt = AGENT_MAP[dept]

        enhanced_query = f"""
PRIMARY TASK:
Answer the following user query:

{latest_query}

---

KNOWLEDGE BASE (Authoritative Source):
{rag_context}

CONTEXT INFO(only for followup questions )
{context}

---

INSTRUCTIONS:
- Base your answer strictly on the provided knowledge base, but use the persona of the detected department to answer the query.
- You may paraphrase and synthesize the information.
- Do NOT introduce facts not present in the knowledge base.
- Use Context Info for your understanding of the conversation and not as replacement of KNOWLEDGE BASE
- If no relevant information exists, say:
  "This information is not available in our records."
- Respond professionally.
"""

        chain = prompt | llm
        result = chain.invoke({"query": enhanced_query})

        print("User Query:", latest_query)
        print("Generated Response:", result.content)

        if "not available in our records" in result.content:
            print(f"⚠️ [System Failure] LLM confirmed no info for {dept}")
        else:
            responses.append(result.content)


    if not any_docs_found or not responses:
        return {
            "responses": ["I'm sorry, I couldn't find any specific information regarding that in our ShopUNow records."],
            "rag_docs_found": False,
            "turn_type": "knowledge_gap",
            "node_name": "execute_departments"

        }

    return {
        "responses": responses,
        "rag_docs_found": True,
        "turn_type": "success",
        "node_name": "execute_departments"
    }



def merge_node(state: ShopState):
    print("Merging the node")

    final = "\n\n".join(state.responses)
    return {"final_response": final}


def response_enrichment_node(state):

    responses = state.responses

    # If only one response, no need to enrich
    if len(responses) <= 1:
        return {"final_response": responses[0] if responses else "", "node_name":"Final Response"}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    combined_responses = "\n\n".join(responses)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a professional response formatting assistant.

Your task:
- Merge multiple department responses into ONE cohesive reply.
- Remove redundancy.
- Ensure smooth flow.
- Maintain professional tone.
- Do NOT mention department names.
- Do NOT say "Department X says".
- Make it look like one unified response.
"""),
        ("human", "{content}")
    ])

    chain = prompt | llm

    enriched = chain.invoke({
        "content": combined_responses
    }).content

    return {"final_response": enriched, "node_name": "final_response"}

def guardrail_block_node(state: ShopState):
        print("Guard Rail block executed")
        return {
            "final_response": "⚠️ Your query is outside the supported scope of this assistant. Please ask about ShopUNow services.",
            "escalation_required": False,
            "node_name": "Guard Rail Block node"
        }

def response_scorer_node(state: ShopState):

    print("Response Scorer Node Executed")

    query = state.optimized_query or state.query
    answer = state.response
    retrieved_docs = state.retrieved_docs  # list of Documents

    # Combine retrieved content for grounding check
    docs_text = "\n\n".join(
        [doc.page_content + "\n" + str(doc.metadata.get("answer", "")) 
         for doc in retrieved_docs]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a strict evaluation agent for ShopUNow responses.

Evaluate the assistant's final response.

Score each metric between 0 and 1.

Metrics:

Relevance:
Does the response directly answer the user's question?

Groundedness:
Is the response supported by retrieved documents?

Completeness:
Does the response fully answer the question?

Hallucination Risk:
Does the response introduce unsupported information?

Clarity:
Is the response clear and understandable?

Compute:
overall_score = average of relevance, groundedness, completeness, clarity minus hallucination_risk impact.

Set needs_regeneration = True if:
- overall_score < 0.6
OR
- hallucination_risk > 0.4

Be strict. Do not inflate scores.

Return structured output only.
"""),
        ("human", """
User Query:
{query}

Retrieved Documents:
{docs}

Assistant Response:
{answer}
""")
    ])

    chain = prompt | llm

    try:
        evaluation = chain.invoke({
            "query": query,
            "docs": docs_text,
            "answer": answer
        })
    except Exception as e:
        print("Scoring error:", e)
        return {"evaluation": None}

    print("Evaluation:", evaluation)

    return {
        "evaluation": evaluation
    }

def answer_grader_node(state: ShopState):
    print("--- ANSWER GRADER NODE ---")
    
    # Initialize LLM with structured output
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(GradeAnswer)
    
    # Prepare context for the judge: The Answer vs. the Source Documents
    # (Assuming state.responses contains the raw department output)
    current_answer = state.final_response or "\n\n".join(state.responses)
    
    system_prompt = """
    You are a Quality Assurance Judge for ShopUNow. 
    Compare the provided ANSWER against the USER QUERY and the retrieved CONTEXT.
    
    - Faithfulness: Ensure the answer does NOT include info outside the context.
    - Relevance: Ensure the answer directly solves the user's specific problem.
    """
    
    result = llm.invoke([
        ("system", system_prompt),
        ("human", f"QUERY: {state.query}\n\nANSWER: {current_answer}")
    ])

    result_dict = result.model_dump() 
    
    print(f"Scores -> Faith: {result.faithfulness_score}, Rel: {result.relevance_score}")
    
    return {
    #"is_satisfactory": result_dict["is_satisfactory"],
    #"improvement_feedback": result_dict["improvement_feedback"],
    "faithfulness_score": result_dict["faithfulness_score"],
    "relevance_score": result_dict["relevance_score"],
    "node_name": "answer_grader"
    }

import logging

def logger_node(state: ShopState):
    """
    Centralized observer node to trace state after every execution step.
    """
    # Identify the last executed node from state or a custom flag
    last_step = state.turn_type or "unknown_node"
    
    print(f"📊 [TRACE] Completed Node: {last_step}")
    logging.info(f"NODE_COMPLETED: {last_step} | Query: {state.query} | Depts: {state.departments}")
    
    if state.rag_docs_found is not None:
        logging.info(f"   RAG Status: {state.rag_docs_found} | Responses: {len(state.responses or [])}")

    # Return empty dict to keep state unchanged
    return {}


def topic_shift_node(state: ShopState):

    print("Topic Shift Detection Node")

    # If we are already waiting for clarification, skip detection
    if state.awaiting_clarification:
        return {
            "topic_shift_detected": False,
            "node_name": "topic_shift"
        }

    if not state.history:
        return {
            "topic_shift_detected": False,
            "node_name": "topic_shift"
        }

    previous_query = state.history[-1].query
    current_departments = state.departments or []

    if not previous_query or not current_departments:
        return {
            "topic_shift_detected": False,
            "node_name": "topic_shift"
        }

    # Use your existing taxonomy logic
    prev_departments = detect_department_from_taxonomy(previous_query)

    if not prev_departments:
        return {
            "topic_shift_detected": False,
            "node_name": "topic_shift"
        }

    # If departments do not overlap → topic shift
    if set(prev_departments).isdisjoint(set(current_departments)):
        print("Topic shift detected.")
        return {
            "topic_shift_detected": True,
            "node_name": "topic_shift"
        }

    return {
        "topic_shift_detected": False,
        "node_name": "topic_shift"
    }

def clarification_node(state: ShopState):

    print("Clarification Node Activated")

    previous_query = state.history[-1].query if state.history else "previous topic"
    current_query = state.query

    response = (
        f"I noticed your previous question was about:\n"
        f"'{previous_query}'.\n\n"
        f"Now you mentioned:\n"
        f"'{current_query}'.\n\n"
        f"Are you switching to this new issue?"
    )

    return {
        "final_response": response,
        "awaiting_clarification": True,
        "clarification_context": {
            "previous_query": previous_query,
            "new_query": current_query
        },
        "node_name": "clarification"
    }