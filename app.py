import streamlit as st
import asyncio
import time
from dotenv import load_dotenv

from graphbuilder import build_graph
from multimodalprocessor import process_uploaded_file
from memory import get_user_history, append_user_history
from config import GraphBusinessLogger

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
# Cleanup Expired Sessions On App Load
# ==========================================================

cleanup_expired_sessions(timeout_minutes=30)

# ==========================================================
# Initialize LangGraph (Only Once)
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

# Create session immediately after phone entry
if "session_id" not in st.session_state:

    session_id = create_user_session(phone)

    st.session_state.session_id = session_id
    st.session_state.session_phone = phone
    st.session_state.session_start_time = time.time()

# If phone changes → close old session & create new one
if (
    "session_phone" in st.session_state
    and st.session_state.session_phone != phone
):
    close_user_session(st.session_state.session_id)

    new_session_id = create_user_session(phone)

    st.session_state.session_id = new_session_id
    st.session_state.session_phone = phone
    st.session_state.session_start_time = time.time()

# Check inactivity timeout (30 mins)
if is_session_expired(st.session_state.session_id, timeout_minutes=30):

    close_user_session(st.session_state.session_id)

    st.warning("⏳ Session expired due to inactivity. Please re-login.")
    st.session_state.clear()
    st.stop()

# ==========================================================
# Session State Initialization
# ==========================================================

if "loaded_phone" not in st.session_state or st.session_state.loaded_phone != phone:
    st.session_state.loaded_phone = phone
    st.session_state.chat_history = get_user_history(phone)
    st.session_state.escalation_active = False

    # Reset clarification state
    st.session_state.awaiting_clarification = False
    st.session_state.clarification_context = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "escalation_active" not in st.session_state:
    st.session_state.escalation_active = False

if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False

if "clarification_context" not in st.session_state:
    st.session_state.clarification_context = None

# ==========================================================
# Display Previous Conversations
# ==========================================================

st.subheader("💬 Conversation")

for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(item["query"])
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

                confirmation_message = {
                    "query": "Support form submitted",
                    "response": "✅ Your support request has been submitted successfully. Our team will contact you shortly."
                }

                st.session_state.chat_history.append(confirmation_message)
                append_user_history(phone, confirmation_message)

                st.session_state.escalation_active = False

                update_session_activity(st.session_state.session_id)

                st.rerun()

    st.stop()

# ==========================================================
# Multimodal File Upload
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

    # Update session activity
    update_session_activity(st.session_state.session_id)

    with st.chat_message("user"):
        st.markdown(user_query)

    combined_query = user_query

    # ======================================================
    # Multimodal Processing
    # ======================================================

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

    # ======================================================
    # Invoke LangGraph
    # ======================================================

    start_time = time.perf_counter()

    with st.spinner("Thinking..."):
        result = asyncio.run(
            graph.ainvoke(
                {
                    "query": combined_query,
                    "phone": phone,
                    "session_id": st.session_state.session_id,
                    "history": st.session_state.chat_history,
                    "sentiment": None,
                    "departments": None,
                    "responses": [],
                    "final_response": None,
                    "escalation_required": st.session_state.escalation_active,
                    "awaiting_clarification": st.session_state.awaiting_clarification,
                    "clarification_context": st.session_state.clarification_context,
                    "has_attachment": bool(uploaded_file),
                }
            )
        )

    end_time = time.perf_counter()
    total_latency = round(end_time - start_time, 3)
    print(f"Total time taken - {total_latency} seconds")

    final_response = result.get("final_response")
    escalation_required = result.get("escalation_required", False)

    st.session_state.awaiting_clarification = result.get(
        "awaiting_clarification",
        False
    )

    st.session_state.clarification_context = result.get(
        "clarification_context"
    )

    if not final_response:
        final_response = "⚠️ Sorry, no response was generated."

    # ======================================================
    # Escalation Handling
    # ======================================================

    if escalation_required:

        st.session_state.escalation_active = True

        escalation_entry = {
            "query": user_query,
            "response": "⚠️ Escalated to human support."
        }

        st.session_state.chat_history.append(escalation_entry)
        append_user_history(phone, escalation_entry)

        st.rerun()

    # ======================================================
    # Normal Response
    # ======================================================

    with st.chat_message("assistant"):
        st.markdown(final_response)

    resolved_query = result.get("query", user_query)

    new_entry = {
        "query": resolved_query,
        "response": final_response
    }

    st.session_state.chat_history.append(new_entry)
    append_user_history(phone, new_entry)