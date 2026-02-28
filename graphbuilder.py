from langgraph.graph import StateGraph, END

from graphstate import ShopState
from graphnodes import (
    guardrail_node,
    guardrail_block_node,
    sentiment_node,
    department_node,
    topic_shift_node,
    clarification_node,
    department_execution_node,
    response_enrichment_node,
    escalation_node,
    answer_grader_node
)

from config import setup_phoenix

setup_phoenix()


def build_graph():

    graph = StateGraph(ShopState)

    # ==========================================================
    # ADD NODES
    # ==========================================================

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("guardrail_block", guardrail_block_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("department", department_node)
    graph.add_node("topic_shift", topic_shift_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("execute_departments", department_execution_node)
    graph.add_node("response_enrichment", response_enrichment_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("answer_grader", answer_grader_node)

    # ==========================================================
    # ENTRY POINT
    # ==========================================================

    graph.set_entry_point("guardrail")

    # ==========================================================
    # GUARDRAIL ROUTING
    # ==========================================================

    def route_after_guardrail(state: ShopState):

        # 🔥 Handle clarification replies FIRST
        if state.awaiting_clarification:
            return "clarification"

        if state.out_of_scope:
            return "guardrail_block"

        return "sentiment"

    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "clarification": "clarification",
            "guardrail_block": "guardrail_block",
            "sentiment": "sentiment"
        }
    )

    # ==========================================================
    # SENTIMENT ROUTING
    # ==========================================================

    def sentiment_router(state: ShopState):

        emotion = state.emotion or "neutral"
        intensity = state.emotion_intensity or 0.0

        if state.escalation_required:
            return "escalation"

        if emotion == "anger" and intensity > 0.7:
            return "escalation"

        if emotion == "fear" and intensity > 0.6:
            return "escalation"

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

    def department_router(state: ShopState):

        if not state.departments:
            return "escalation"

        return "topic_shift"

    graph.add_conditional_edges(
        "department",
        department_router,
        {
            "escalation": "escalation",
            "topic_shift": "topic_shift"
        }
    )

    # ==========================================================
    # TOPIC SHIFT ROUTING
    # ==========================================================

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
    # CLARIFICATION ROUTING
    # ==========================================================

    def clarification_router(state):

        if state.awaiting_clarification:
            return "END"

        if state.clarification_resolved == "reject":
            return "END"

        if state.clarification_resolved == "confirm":
            return "department"

        return "END"

    graph.add_conditional_edges(
        "clarification",
        clarification_router,
        {
            "department": "department",
            "END": END
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