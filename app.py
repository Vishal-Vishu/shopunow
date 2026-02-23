import streamlit as st
from graphbuilder import build_graph
from multimodalprocessor import process_uploaded_file
from memory import get_user_history, append_user_history
from support_db import save_support_ticket
from config import GraphBusinessLogger
import asyncio

from dotenv import load_dotenv
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
# Session State Initialization
# ==========================================================

if "loaded_phone" not in st.session_state or st.session_state.loaded_phone != phone:
    st.session_state.loaded_phone = phone
    st.session_state.chat_history = get_user_history(phone)
    st.session_state.escalation_active = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "escalation_active" not in st.session_state:
    st.session_state.escalation_active = False


# ==========================================================
# Display Previous Conversations
# ==========================================================

st.subheader("💬 Conversation")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================================
# Reset Button
# ==========================================================

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔄 Reset Chat"):
        st.session_state.chat_history = []
        st.session_state.escalation_active = False
        st.rerun()


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
                # Find last user message safely
                last_user_message = None
                for msg in reversed(st.session_state.chat_history):
                    if msg["role"] == "user":
                        last_user_message = msg["content"]
                        break

                save_support_ticket({
                    "phone": phone,
                    "name": name,
                    "email": email,
                    "issue": issue_details,
                    "original_query": last_user_message
                })

                confirmation_text = (
                    "✅ Your support request has been submitted successfully. "
                    "Our team will contact you shortly."
                )

                # Update session state
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": confirmation_text}
                )

                # Store in DB
                append_user_history(phone, {
                    "query": "Support form submitted",
                    "response": confirmation_text
                })

                st.session_state.escalation_active = False
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

    # Display user message immediately
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

    with st.spinner("Thinking..."):
        result = asyncio.run(graph.ainvoke({
            "query": combined_query,
            "phone": phone,
            "history": st.session_state.chat_history,
            "sentiment": None,
            "departments": None,
            "responses": [],
            "final_response": None,
            "escalation_required": False
        }, config={"callbacks": [GraphBusinessLogger()]}))

    final_response = result.get("final_response")
    escalation_required = result.get("escalation_required", False)

    if not final_response:
        final_response = "⚠️ Sorry, no response was generated."

    # ======================================================
    # Escalation Handling
    # ======================================================

    if escalation_required:

        st.session_state.escalation_active = True

        escalation_text = "⚠️ Escalated to human support."

        # Update session state
        st.session_state.chat_history.append(
            {"role": "user", "content": user_query}
        )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": escalation_text}
        )

        # Store in DB
        append_user_history(phone, {
            "query": user_query,
            "response": escalation_text
        })

        st.rerun()

    # ======================================================
    # Normal Response
    # ======================================================

    with st.chat_message("assistant"):
        st.markdown(final_response)

    # Update session state
    st.session_state.chat_history.append(
        {"role": "user", "content": user_query}
    )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": final_response}
    )

    # Store in DB
    append_user_history(phone, {
        "query": user_query,
        "response": final_response
    })