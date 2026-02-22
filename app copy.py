import streamlit as st
from graphbuilder import build_graph
from multimodalprocessor import process_uploaded_file
from memory import get_user_history, append_user_history


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
# Session State Handling (Critical for Memory Sync)
# ==========================================================

if "loaded_phone" not in st.session_state or st.session_state.loaded_phone != phone:
    st.session_state.loaded_phone = phone
    st.session_state.chat_history = get_user_history(phone)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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
# Reset Button
# ==========================================================

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔄 Reset Chat"):
        st.session_state.chat_history = []
        st.rerun()


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

    # Display user message instantly
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
    with st.spinner("Thinking..."):
        result = graph.invoke({
            "query": combined_query,
            "phone": phone,
            "history": st.session_state.chat_history,
            "sentiment": None,
            "departments": None,
            "responses": [],
            "final_response": None
        })

    escalation_required = result.get("escalation_required", False)    

    final_response = result.get("final_response")

    if "escalation_mode" not in st.session_state:
        st.session_state.escalation_mode = False

    if result.get("escalation_required"):
        st.session_state.escalation_mode = True        


    if not final_response:
        final_response = "⚠️ Sorry, no response was generated."

    if st.session_state.escalation_mode:
        st.warning("This issue requires assistance from our human support team.")

        with st.form("escalation_form", clear_on_submit=True):
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            issue_details = st.text_area("Describe your issue in detail")

            submit = st.form_submit_button("Submit to Support")
            print("User esclation form submission")
            print(submit)

    if submit:
                print("form submission for escalation")
                from database import save_escalation
                save_escalation(name, phone, email, issue_details)

                st.session_state.escalation_mode = False
                st.session_state.escalation_submitted = True

                st.rerun()

                st.success("✅ Your request has been submitted successfully. Our support team will contact you shortly.")
    else:
                with st.chat_message("assistant"):
                    st.markdown(result.get("final_response", ""))

    with st.chat_message("assistant"):
        st.markdown(final_response)

    new_entry = {
        "query": user_query,
        "response": final_response
    }

    st.session_state.chat_history.append(new_entry)

    # Persist to JSON storage
    append_user_history(phone, new_entry)
