import streamlit as st
import asyncio
import time
from dotenv import load_dotenv

from graphbuilder import build_graph
from multimodalprocessor import process_uploaded_file

from memory import (
    initialize_conversation_table,
    append_message,
    fetch_session_history
)

from support_db import (
    save_support_ticket,
    create_user_session,
    update_session_activity,
    close_user_session,
    is_session_expired,
    cleanup_expired_sessions
)

load_dotenv()

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="ShopUNow Agentic AI Assistant",
    layout="wide"
)

st.title("🛍️ ShopUNow Agentic AI Assistant")

# ==========================================================
# Cleanup Expired Sessions
# ==========================================================

cleanup_expired_sessions(timeout_minutes=30)

# ==========================================================
# Initialize Conversation Table (DB Memory)
# ==========================================================

initialize_conversation_table()

# ==========================================================
# Initialize Graph (Cached)
# ==========================================================

@st.cache_resource
def initialize_graph():
    return build_graph()

graph = initialize_graph()

# ==========================================================
# Mobile Login
# ==========================================================

phone = st.text_input("📱 Enter Mobile Number")

if not phone:
    st.warning("Please enter your mobile number to continue.")
    st.stop()

# ==========================================================
# SESSION MANAGEMENT
# ==========================================================

if "session_id" not in st.session_state:
    session_id = create_user_session(phone)
    st.session_state.session_id = session_id
    st.session_state.session_phone = phone
    st.session_state.session_start_time = time.time()

# If phone changes
if (
    "session_phone" in st.session_state
    and st.session_state.session_phone != phone
):
    close_user_session(st.session_state.session_id)

    new_session_id = create_user_session(phone)
    st.session_state.session_id = new_session_id
    st.session_state.session_phone = phone
    st.session_state.session_start_time = time.time()

# Session timeout
if is_session_expired(st.session_state.session_id, timeout_minutes=30):
    close_user_session(st.session_state.session_id)
    st.warning("⏳ Session expired due to inactivity. Please re-login.")
    st.session_state.clear()
    st.stop()

# ==========================================================
# Load Conversation History (Session-Based)
# ==========================================================

if (
    "loaded_session" not in st.session_state
    or st.session_state.loaded_session != st.session_state.session_id
):
    st.session_state.loaded_session = st.session_state.session_id
    st.session_state.chat_history = fetch_session_history(
        st.session_state.session_id,
        limit=20
    )
    st.session_state.escalation_active = False
    st.session_state.awaiting_clarification = False
    st.session_state.clarification_context = None

# ==========================================================
# Display Conversation
# ==========================================================

st.subheader("💬 Conversation")

for item in st.session_state.chat_history:
    if item.get("query"):
        with st.chat_message("user"):
            st.markdown(item["query"])
    if item.get("response"):
        with st.chat_message("assistant"):
            st.markdown(item["response"])

# ==========================================================
# Escalation Form
# ==========================================================

if st.session_state.escalation_active:

    st.warning("⚠️ Your issue requires assistance from a human support agent.")

    with st.form("support_form"):
        st.subheader("📝 Contact Support")

        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        issue_details = st.text_area("Describe your issue in detail")

        submitted = st.form_submit_button("Submit")

        if submitted:
            if not name or not email or not issue_details:
                st.error("Please complete all fields.")
            else:
                save_support_ticket({
                    "phone": phone,
                    "name": name,
                    "email": email,
                    "issue": issue_details,
                    "original_query": st.session_state.chat_history[-1]["query"]
                })

                append_message(
                    phone=phone,
                    session_id=st.session_state.session_id,
                    query="Support form submitted",
                    response="✅ Your support request has been submitted successfully. Our team will contact you shortly.",
                    turn_type="escalation_confirmation",
                    department="HUMAN_SUPPORT",
                    sentiment="neutral",
                    emotion="neutral"
                )

                st.session_state.chat_history = fetch_session_history(
                    st.session_state.session_id,
                    limit=20
                )

                st.session_state.escalation_active = False
                update_session_activity(st.session_state.session_id)
                st.rerun()

    st.stop()

# ==========================================================
# File Upload
# ==========================================================

uploaded_file = st.file_uploader(
    "📎 Upload bill / receipt / image / document (optional)",
    type=["png", "jpg", "jpeg", "pdf", "docx"]
)

# ==========================================================
# Chat Input
# ==========================================================

user_query = st.chat_input("Type your message here...")

if user_query:

    update_session_activity(st.session_state.session_id)

    combined_query = user_query

    # --------------------------------------
    # Multimodal Processing
    # --------------------------------------

    if uploaded_file:
        with st.spinner("Processing uploaded file..."):
            extracted_text = process_uploaded_file(uploaded_file)

        with st.expander("📄 Extracted File Content"):
            st.text(extracted_text)

        combined_query = f"""
User Query:
{user_query}

Attached Document Content:
{extracted_text}

Please consider both while responding.
"""

    # --------------------------------------
    # Graph Invocation
    # --------------------------------------

    with st.spinner("Thinking..."):
        result = asyncio.run(
            graph.ainvoke(
                {
                    "query": combined_query,
                    "phone": phone,
                    "session_id": st.session_state.session_id,
                    "history": fetch_session_history(
                        st.session_state.session_id,
                        limit=20
                    ),
                    "sentiment": None,
                    "departments": None,
                    "responses": [],
                    "final_response": None,
                    "escalation_required": False,
                    "awaiting_clarification": st.session_state.awaiting_clarification,
                    "clarification_context": st.session_state.clarification_context,
                    "has_attachment": bool(uploaded_file),
                }
            )
        )

    final_response = result.get("final_response") or "⚠️ No response generated."
    escalation_required = result.get("escalation_required", False)

    st.session_state.awaiting_clarification = result.get(
        "awaiting_clarification", False
    )

    st.session_state.clarification_context = result.get(
        "clarification_context"
    )

    # --------------------------------------
    # Escalation Trigger
    # --------------------------------------

    if escalation_required:

        append_message(
            phone=phone,
            session_id=st.session_state.session_id,
            query=user_query,
            response="⚠️ Escalated to human support.",
            turn_type="escalation_triggered",
            department="HUMAN_SUPPORT",
            sentiment=result.get("sentiment"),
            emotion=result.get("emotion")
        )

        st.session_state.chat_history = fetch_session_history(
            st.session_state.session_id,
            limit=20
        )

        st.session_state.escalation_active = True
        st.rerun()

    # --------------------------------------
    # Normal Response Logging
    # --------------------------------------

    append_message(
        phone=phone,
        session_id=st.session_state.session_id,
        query=result.get("query", user_query),
        response=final_response,
        turn_type=result.get("turn_type", "success"),
        department=",".join(result.get("departments", [])) if result.get("departments") else None,
        sentiment=result.get("sentiment"),
        emotion=result.get("emotion")
    )

    st.session_state.chat_history = fetch_session_history(
        st.session_state.session_id,
        limit=20
    )

    st.rerun()