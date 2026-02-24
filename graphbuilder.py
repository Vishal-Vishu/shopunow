from langgraph.graph import StateGraph, END

from graphstate import ShopState
from graphnodes import (
    preprocess_node,
    sentiment_node,
    department_execution_node,
    response_enrichment_node,
    escalation_node,
    department_node,
    guardrail_node,
    guardrail_block_node,
    rewrite_node,
    answer_grader_node,
    topic_shift_node,
    clarification_node
)

from config import setup_phoenix

setup_phoenix()

def node_tracer(event):
    # This captures the output of every node automatically
    node_name = event['metadata'].get('langgraph_node')
    if node_name:
        print(f"DEBUG: Node {node_name} just finished.")

def build_graph():

    graph = StateGraph(ShopState)

    # ==========================================================
    # ADD NODES
    # ==========================================================

    

   # graph.add_node("preprocess", preprocess_node)              # Non-LLM safety + init
    graph.add_node("rewrite", rewrite_node)                    # LLM normalization
    graph.add_node("guardrail", guardrail_node)                # Semantic safety check
    graph.add_node("guardrail_block", guardrail_block_node)    # Block response
    graph.add_node("sentiment", sentiment_node)                # Sentiment analysis
    graph.add_node("department", department_node)              # Department routing
    graph.add_node("execute_departments", department_execution_node)
    #graph.add_node("merge", merge_node)
    graph.add_node("response_enrichment", response_enrichment_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("answer_grader", answer_grader_node)
    graph.add_node("topic_shift", topic_shift_node)
    graph.add_node("clarification", clarification_node)

    # ==========================================================
    # ENTRY POINT
    # ==========================================================

    graph.set_entry_point("rewrite")

    # ==========================================================
    # PREPROCESS → REWRITE
    # ==========================================================

    #graph.add_edge("preprocess", "rewrite")

    # ==========================================================
    # REWRITE → GUARDRAIL
    # ==========================================================

    graph.add_edge("rewrite", "guardrail")

    # ==========================================================
    # GUARDRAIL ROUTING
    # ==========================================================

    def route_after_guardrail(state: ShopState):
        """
        If query is out of scope or high risk → block.
        Otherwise continue to sentiment analysis.
        """
        if state.out_of_scope:
            return "guardrail_block"
        return "sentiment"

    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "guardrail_block": "guardrail_block",
            "sentiment": "sentiment"
        }
    )

    # ==========================================================
    # SENTIMENT ROUTING
    # ==========================================================

    def sentiment_router(state: ShopState):
        """
        Emotion-aware routing logic.
        """

        # Safety fallback
        emotion = state.emotion or "neutral"
        intensity = state.emotion_intensity or 0.0

        # 1️⃣ Immediate escalation conditions
        if state.escalation_required:
            return "escalation"

        # 2️⃣ Strong anger → escalate
        if emotion == "anger" and intensity > 0.7:
            return "escalation"

        # 3️⃣ Fear → escalate (trust risk)
        if emotion == "fear" and intensity > 0.6:
            return "escalation"

        # 4️⃣ Sadness or disappointment → allow department handling
        # but with emotional tone injection (already implemented)

        # 5️⃣ Confusion → continue to department (likely needs explanation)

        return "department"

    graph.add_conditional_edges(
        "sentiment",
        sentiment_router,
        {
            "escalation": "escalation",
            "department": "department"
        }
    )

    # ==========================================================
    # DEPARTMENT ROUTING
    # ==========================================================

    def department_router_fn(state: ShopState):
        """
        If no department identified → escalate.
        Otherwise execute department handlers.
        """
        if not state.departments:
            return "escalation"
        return "execute_departments"

    graph.add_conditional_edges(
        "department",
        department_router_fn,
        {
            "escalation": "escalation",
            "execute_departments": "topic_shift"
        }
    )

    def topic_shift_router(state: ShopState):
        if state.topic_shift_detected:
            return "clarification"
        return "execute_departments"

    graph.add_conditional_edges(
        "topic_shift",
        topic_shift_router,
        {
            "clarification": "clarification",
            "execute_departments": "execute_departments"
        }
    )

    # ==========================================================
    # EXECUTION FLOW
    # ==========================================================

    graph.add_edge("execute_departments", "response_enrichment")
    graph.add_edge("response_enrichment", "answer_grader")

    # ==========================================================
    # TERMINAL STATES
    # ==========================================================

    graph.add_edge("response_enrichment", END)
    graph.add_edge("guardrail_block", END)
    graph.add_edge("escalation", END)

    return graph.compile()


# ==========================================================
# EXPORT GRAPH STRUCTURE (Optional)
# ==========================================================

from pathlib import Path

if __name__ == "__main__":

    graph = build_graph()

    diagram_path = Path("langgraph_structure.md")

    with open(diagram_path, "w") as f:
        f.write("```mermaid\n")
        f.write(graph.get_graph().draw_mermaid())
        f.write("\n```")

    print("LangGraph diagram exported to langgraph_structure.md")
