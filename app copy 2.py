import streamlit as st
from graphbuilder import build_graph
from multimodalprocessor import process_uploaded_file
from memory import get_user_history, append_user_history
from database import init_db, save_escalation


# ==========================================================
# INITIAL SETUP
# ==========================================================

st.set_page_config(page_title="ShopUNow Agentic AI Assistant", layout="wide")
st.title("🛍️ ShopUNow Agentic AI Assistant")

init_db()

# Initialize graph only once
@st.cache_resource
def initialize_graph():
    return build_graph()

graph = initialize_graph()


# ==========================================================
# MOBILE LOGIN
# ==========================================================

phone = st.text_input("📱 Enter Mobile Number")

if not phone:
    st.warning("Please enter your mobile number to continue.")
    st.stop()


# ==========================================================
# SESSION STATE INITIALIZATION
# ==========================================================

if "loaded_phone" not in st.session_state or st.session_state.loaded_phone != phone:
    st.session_state.loaded_phone = phone
    st.session_state.chat_history = get_user_history(phone)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "escalation_mode" not in st.session_state:
    st.session_state.escalation_mode = False

if "escalation_submitted" not in st.session_state:
    st.session_state.escalation_submitted = False


# ==========================================================
# DISPLAY CHAT HISTORY
# ==========================================================

st.subheader("💬 Conversation")

for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(item["query"])
    with st.chat_message("assistant"):
        st.markdown(item["response"])


# ==========================================================
# RESET CHAT
# ==========================================================

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🔄 Reset Chat"):
        st.session_state.chat_history = []
        st.session_state.escalation_mode = False
        st.session_state.escalation_submitted = False
        st.rerun()


# ==========================================================
# FILE UPLOAD (OPTIONAL)
# ==========================================================

uploaded_file = st.file_uploader(
    "📎 Upload bill / receipt / image / document (optional)",
    type=["png", "jpg", "jpeg", "pdf", "docx"]
)


# ==========================================================
# CHAT INPUT
# ==========================================================

user_query = st.chat_input("Type your message here...")


if user_query:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    combined_query = user_query

    # Multimodal processing
    if uploaded_file:
        with st.spinner("Processing uploaded file..."):
            extracted_text = process_uploaded_file(uploaded_file)

        combined_query += f"\n\nAttached Document:\n{extracted_text}"

    # Invoke graph
    with st.spinner("Thinking..."):
        result = graph.invoke({
            "query": combined_query,
            "phone": phone,
            "history": st.session_state.chat_history
        })

    final_response = result.get("final_response", "")
    escalation_required = result.get("escalation_required", False)

    

    if escalation_required:
        st.session_state.escalation_mode = True
        st.session_state.escalation_submitted = False
    else:
        with st.chat_message("assistant"):
            st.session_state.escalation_mode = False
            st.markdown(final_response)

        # Save to history
        new_entry = {
            "query": user_query,
            "response": final_response
        }

        st.session_state.chat_history.append(new_entry)
        append_user_history(phone, new_entry)


# ==========================================================
# ESCALATION FORM (PERSISTENT)
# ==========================================================

if st.session_state.escalation_mode:

    st.warning("⚠️ This issue requires assistance from our human support team.")

    with st.form("escalation_form", clear_on_submit=True):

        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        issue_details = st.text_area("Describe your issue in detail")

        submit = st.form_submit_button("Submit to Support")
        print("Submit")
        print(submit)
        if submit:

            save_escalation(name, phone, email, issue_details)

            st.session_state.escalation_mode = False
            st.session_state.escalation_submitted = True

            st.rerun()


# ==========================================================
# ESCALATION CONFIRMATION
# ==========================================================

if st.session_state.escalation_submitted:
    st.success("✅ Your request has been submitted successfully. Our support team will contact you shortly.")
