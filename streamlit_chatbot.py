"""
ShopUNow AI Assistant - Streamlit Chatbot Interface with Multi-Modal Support
=============================================================================
Features:
- Beautiful chatbot interface
- Image upload support (receipts, bills, photos)
- PDF upload support (invoices, reports)
- Document analysis with GPT-4 Vision
- Multi-user conversation history

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import uuid
from datetime import datetime
import json
import os
from PIL import Image
import io

# Import the agent and database manager
from shopagent import (
    ShopUNowEnhancedAgent, 
    DEPARTMENTS_CONFIG,
    EscalationForm,
    DocumentProcessor
)
from databasemanager import DatabaseManager

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="ShopUNow AI Assistant",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: white;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f8f9fa;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1 {
        color: white;
        text-align: center;
        padding: 20px;
        background: rgba(0,0,0,0.2);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    h2, h3 {
        color: #667eea;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Text input */
    .stTextInput>div>div>input {
        border-radius: 25px;
        border: 2px solid #667eea;
        padding: 10px 20px;
    }
    
    /* Metrics */
    .css-1xarl3l {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Info boxes */
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Department badges */
    .dept-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        margin: 5px;
        font-size: 0.9em;
    }
    
    /* Escalation warning */
    .escalation-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Success message */
    .success-message {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager("shopunow_assistant.db")
    
    if 'user_authenticated' not in st.session_state:
        st.session_state.user_authenticated = False
    
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'conversation_stats' not in st.session_state:
        st.session_state.conversation_stats = {
            'total_queries': 0,
            'escalations': 0,
            'departments_used': set()
        }
    
    if 'show_escalation_form' not in st.session_state:
        st.session_state.show_escalation_form = False
    
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False
    
    if 'uploaded_files_data' not in st.session_state:
        st.session_state.uploaded_files_data = []
    
    if 'current_files' not in st.session_state:
        st.session_state.current_files = []
    
    if 'history_loaded' not in st.session_state:
        st.session_state.history_loaded = False

# ============================================================================
# AGENT INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_agent(api_key):
    """Initialize the AI agent (cached to prevent reloading)"""
    try:
        agent = ShopUNowEnhancedAgent(api_key, enable_email=False)
        agent.create_vector_database()
        agent.graph = agent.build_graph()
        return agent, None
    except Exception as e:
        return None, str(e)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_message(role, content, metadata=None):
    """Add a message to the chat history and database"""
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'metadata': metadata or {}
    }
    st.session_state.messages.append(message)
    
    # Save to database if user is authenticated
    if st.session_state.user_authenticated and st.session_state.user_info:
        try:
            st.session_state.db.save_message(
                user_id=st.session_state.user_info['id'],
                session_id=st.session_state.session_id,
                role=role,
                content=content,
                departments=metadata.get('departments', []),
                sentiment=metadata.get('sentiment'),
                escalated=metadata.get('escalated', False),
                has_attachments=metadata.get('has_files', False)
            )
        except Exception as e:
            print(f"Error saving message to DB: {e}")

def display_message(message):
    """Display a single chat message with styling"""
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
        # Display metadata if available
        if message['metadata']:
            metadata = message['metadata']
            
            # Display departments
            if 'departments' in metadata and metadata['departments']:
                dept_html = "**Departments:** "
                for dept in metadata['departments']:
                    dept_name = DEPARTMENTS_CONFIG.get(dept, {}).get('name', dept)
                    dept_html += f'<span class="dept-badge">{dept_name}</span> '
                st.markdown(dept_html, unsafe_allow_html=True)
            
            # Display sentiment
            if 'sentiment' in metadata:
                sentiment = metadata['sentiment']
                emoji = "😊" if sentiment == "positive" else "😐" if sentiment == "neutral" else "😟"
                st.caption(f"Sentiment: {emoji} {sentiment.title()}")
            
            # Display escalation status
            if metadata.get('escalated'):
                st.markdown(
                    '<div class="escalation-warning">⚠️ This query has been escalated to human support</div>',
                    unsafe_allow_html=True
                )

def process_query(query, escalation_form_data=None, uploaded_files=None):
    """Process user query through the agent with optional file uploads"""
    if not st.session_state.agent:
        return "❌ Agent not initialized. Please check your API key."
    
    try:
        # Show processing indicator
        with st.spinner("🤔 Analyzing your query and documents..."):
            result = st.session_state.agent.query(
                user_id=st.session_state.user_id,
                user_query=query,
                session_id=st.session_state.session_id,
                escalation_form=escalation_form_data,
                uploaded_files=uploaded_files
            )
        
        # Update stats
        st.session_state.conversation_stats['total_queries'] += 1
        if result['escalated']:
            st.session_state.conversation_stats['escalations'] += 1
        for dept in result['departments_identified']:
            st.session_state.conversation_stats['departments_used'].add(dept)
        
        return result
    
    except Exception as e:
        return {"response": f"❌ Error: {str(e)}", "error": True}

def process_uploaded_files(uploaded_files):
    """Process uploaded files for agent consumption"""
    processed_files = []
    
    for file in uploaded_files:
        try:
            file_bytes = file.read()
            file_type = DocumentProcessor.get_file_type(file.name)
            
            processed_files.append({
                "filename": file.name,
                "type": file_type,
                "data": file_bytes,
                "size": len(file_bytes)
            })
            
            # Reset file pointer
            file.seek(0)
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    return processed_files

def clear_conversation():
    """Clear the conversation history and end session"""
    # End current session in database
    if st.session_state.user_authenticated:
        departments_list = list(st.session_state.conversation_stats['departments_used'])
        st.session_state.db.end_session(
            st.session_state.session_id,
            departments_list
        )
    
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.conversation_stats = {
        'total_queries': 0,
        'escalations': 0,
        'departments_used': set()
    }
    st.session_state.history_loaded = False
    
    # Create new session in database
    if st.session_state.user_authenticated and st.session_state.user_info:
        st.session_state.db.create_session(
            st.session_state.user_info['id'],
            st.session_state.session_id
        )

def load_conversation_history():
    """Load user's previous conversations from database"""
    if not st.session_state.user_authenticated or not st.session_state.user_info:
        return
    
    if st.session_state.history_loaded:
        return
    
    try:
        # Get last 10 messages from previous sessions
        history = st.session_state.db.get_conversation_history(
            st.session_state.user_info['id'],
            limit=10
        )
        
        if history:
            st.session_state.messages = []
            # Reverse to show oldest first
            for msg in reversed(history):
                st.session_state.messages.append({
                    'role': msg['role'],
                    'content': msg['content'],
                    'timestamp': msg['timestamp'],
                    'metadata': {
                        'departments': msg['departments'],
                        'sentiment': msg['sentiment'],
                        'escalated': msg['escalated']
                    }
                })
        
        st.session_state.history_loaded = True
    except Exception as e:
        print(f"Error loading history: {e}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    initialize_session_state()
    
    # ========================================================================
    # AUTHENTICATION SCREEN
    # ========================================================================
    
    if not st.session_state.user_authenticated:
        st.markdown("<h1 style='text-align: center;'>🛍️ ShopUNow AI Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #667eea;'>Welcome! Please sign in to continue</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div class="info-box" style="margin-top: 50px;">
                <h3>📱 Sign In with Mobile Number</h3>
                <p>Enter your mobile number to access your personalized AI assistant.</p>
                <ul>
                    <li>🔒 Secure and private</li>
                    <li>💾 Your conversations are saved</li>
                    <li>📊 Track your history</li>
                    <li>⚡ Quick access to past queries</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("auth_form"):
                st.markdown("### Enter Your Details")
                
                mobile_number = st.text_input(
                    "Mobile Number *",
                    placeholder="+1234567890 or 9876543210",
                    help="Enter your 10-15 digit mobile number with optional country code"
                )
                
                name = st.text_input(
                    "Name (Optional)",
                    placeholder="John Doe",
                    help="Your name for personalized assistance"
                )
                
                email = st.text_input(
                    "Email (Optional)",
                    placeholder="john@example.com",
                    help="Email for important notifications"
                )
                
                user_type = st.selectbox(
                    "I am a:",
                    ["Customer", "Employee"],
                    help="Select your user type for relevant departments"
                )
                
                submit = st.form_submit_button("🚀 Start Chat")
                
                if submit:
                    if not mobile_number:
                        st.error("❌ Please enter your mobile number")
                    else:
                        try:
                            # Validate and create/get user
                            user = st.session_state.db.create_or_get_user(
                                mobile_number=mobile_number,
                                name=name if name else None,
                                email=email if email else None,
                                user_type=user_type.lower()
                            )
                            
                            st.session_state.user_info = user
                            st.session_state.user_id = f"user_{user['id']}"
                            st.session_state.user_authenticated = True
                            
                            # Create session in database
                            st.session_state.db.create_session(
                                user['id'],
                                st.session_state.session_id
                            )
                            
                            # Show welcome message
                            if user['is_new']:
                                st.success(f"✅ Welcome, {user['name'] or 'there'}! Your account has been created.")
                            else:
                                st.success(f"✅ Welcome back, {user['name'] or 'there'}! Loading your conversation history...")
                            
                            st.balloons()
                            st.rerun()
                            
                        except ValueError as e:
                            st.error(f"❌ {str(e)}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            st.markdown("""
            <div style="text-align: center; margin-top: 30px; color: #666;">
                <p>🔒 Your data is secure and encrypted</p>
                <p style="font-size: 0.8em;">By continuing, you agree to our Terms of Service and Privacy Policy</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # ========================================================================
    # MAIN CHAT INTERFACE (Only shown after authentication)
    # ========================================================================
    
    # Load conversation history on first load
    if not st.session_state.history_loaded:
        load_conversation_history()
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=ShopUNow")
        
        # User Profile Section
        st.markdown("## 👤 User Profile")
        user = st.session_state.user_info
        
        st.markdown(f"""
        <div class="info-box">
            <p><strong>📱 Mobile:</strong> {user['mobile_number']}</p>
            <p><strong>👤 Name:</strong> {user['name'] or 'Not provided'}</p>
            <p><strong>📧 Email:</strong> {user['email'] or 'Not provided'}</p>
            <p><strong>👥 Type:</strong> {user['user_type'].title()}</p>
            <p><strong>📅 Member since:</strong> {user['created_at'][:10]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get user stats from database
        try:
            db_stats = st.session_state.db.get_user_stats(user['id'])
            
            st.markdown("### 📊 Your Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Sessions", db_stats['total_sessions'])
            with col2:
                st.metric("Total Messages", db_stats['total_messages'])
            
            st.metric("Escalations", db_stats['escalation_count'])
            
            if db_stats['departments_used']:
                st.markdown("**Departments Used:**")
                for dept in db_stats['departments_used']:
                    dept_name = DEPARTMENTS_CONFIG.get(dept, {}).get('name', dept)
                    st.markdown(f"- {dept_name}")
        except Exception as e:
            st.warning(f"Could not load stats: {e}")
        
        st.markdown("---")
        
        st.markdown("## ⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )
        
        # Initialize agent button
        if st.button("🚀 Initialize Agent"):
            if not api_key:
                st.error("Please enter your OpenAI API key")
            else:
                with st.spinner("Initializing agent..."):
                    agent, error = initialize_agent(api_key)
                    if error:
                        st.error(f"Initialization failed: {error}")
                    else:
                        st.session_state.agent = agent
                        st.session_state.agent_initialized = True
                        st.success("✅ Agent initialized successfully!")
                        st.rerun()
        
        # Agent status
        if st.session_state.agent_initialized:
            st.success("✅ Agent Active")
        else:
            st.warning("⚠️ Agent Not Initialized")
        
        st.markdown("---")
        
        # Session info
        st.markdown("## 📍 Current Session")
        st.text(f"Session: {st.session_state.session_id[:16]}...")
        
        st.markdown("---")
        
        # Statistics
        st.markdown("## 📈 Session Stats")
        stats = st.session_state.conversation_stats
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", stats['total_queries'])
        with col2:
            st.metric("Escalations", stats['escalations'])
        
        st.metric("Departments", len(stats['departments_used']))
        
        if stats['departments_used']:
            st.markdown("**This Session:**")
            for dept in stats['departments_used']:
                dept_name = DEPARTMENTS_CONFIG.get(dept, {}).get('name', dept)
                st.markdown(f"- {dept_name}")
        
        st.markdown("---")
        
        # Department Information
        with st.expander("🏢 Available Departments"):
            st.markdown("### External (Customers)")
            for dept_id, info in DEPARTMENTS_CONFIG.items():
                if info['type'] == 'external':
                    st.markdown(f"**{info['name']}**")
                    st.caption(info['description'])
            
            st.markdown("### Internal (Employees)")
            for dept_id, info in DEPARTMENTS_CONFIG.items():
                if info['type'] == 'internal':
                    st.markdown(f"**{info['name']}**")
                    st.caption(info['description'])
        
        st.markdown("---")
        
        # Action buttons
        if st.button("🔄 New Conversation"):
            clear_conversation()
            st.rerun()
        
        if st.button("📥 Download Chat History"):
            chat_history = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                label="💾 Download JSON",
                data=chat_history,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Logout button
        if st.button("🚪 Logout", type="primary"):
            # End session
            departments_list = list(st.session_state.conversation_stats['departments_used'])
            st.session_state.db.end_session(
                st.session_state.session_id,
                departments_list
            )
            
            # Clear authentication
            st.session_state.user_authenticated = False
            st.session_state.user_info = None
            st.session_state.user_id = None
            st.session_state.messages = []
            st.session_state.history_loaded = False
            
            st.success("✅ Logged out successfully!")
            st.rerun()
    
    # ========================================================================
    # MAIN CHAT INTERFACE
    # ========================================================================
    
    st.markdown("<h1>🛍️ ShopUNow AI Assistant</h1>", unsafe_allow_html=True)
    
    # Welcome message
    if not st.session_state.messages:
        st.markdown("""
        <div class="info-box">
            <h3>👋 Welcome to ShopUNow AI Assistant!</h3>
            <p>I'm here to help you with:</p>
            <ul>
                <li>🛒 Customer Service - Orders, Returns, Shipping</li>
                <li>📦 Product Information - Product details, Reviews, Specifications</li>
                <li>💰 Sales & Promotions - Current deals, Discounts, Loyalty programs</li>
                <li>🔧 Technical Support - Product troubleshooting, Setup help</li>
                <li>👥 HR Support (Employees) - Benefits, Leave, Payroll</li>
                <li>💻 IT Support (Employees) - Technical issues, Access requests</li>
                <li>💵 Finance (Employees) - Expense reports, Reimbursements</li>
                <li>🚚 Operations (Employees) - Inventory, Logistics, Warehouse</li>
            </ul>
            <p><strong>🎨 Multi-Modal Support:</strong></p>
            <ul>
                <li>📸 Upload receipts, invoices, bills</li>
                <li>🖼️ Upload product photos (damage, defects)</li>
                <li>📄 Upload PDFs (expense reports, documents)</li>
                <li>💬 Ask questions about your uploaded files!</li>
            </ul>
            <p><strong>I can handle multi-department queries too!</strong> For example: "What sales do you have on laptops and what's the return policy?"</p>
        </div>
        """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📎 Upload Documents (Optional)")
    
    uploaded_files = st.file_uploader(
        "Upload images, PDFs, or documents",
        type=['jpg', 'jpeg', 'png', 'pdf', 'gif', 'bmp'],
        accept_multiple_files=True,
        help="Upload receipts, invoices, bills, product photos, expense reports, or any relevant documents",
        key="file_uploader"
    )
    
    # Display uploaded files
    if uploaded_files:
        st.markdown("**📁 Uploaded Files:**")
        cols = st.columns(min(len(uploaded_files), 4))
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 4]:
                file_type = DocumentProcessor.get_file_type(file.name)
                
                if file_type == "image":
                    # Display image thumbnail
                    image = Image.open(file)
                    st.image(image, caption=file.name)
                    file.seek(0)  # Reset pointer
                elif file_type == "pdf":
                    st.markdown(f"📄 {file.name}")
                else:
                    st.markdown(f"📎 {file.name}")
                
                st.caption(f"Size: {len(file.read())/1024:.1f} KB")
                file.seek(0)  # Reset pointer
        
        # Store in session state
        st.session_state.current_files = uploaded_files
    else:
        st.session_state.current_files = []
    
    # Display chat messages
    for message in st.session_state.messages:
        display_message(message)
    
    # Escalation form (if triggered)
    if st.session_state.show_escalation_form:
        st.markdown("""
        <div class="escalation-warning">
            <h3>⚠️ Escalation Required</h3>
            <p>To better assist you, please provide your contact information:</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("escalation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name *", placeholder="John Doe")
                email = st.text_input("Email *", placeholder="john@example.com")
            
            with col2:
                phone = st.text_input("Phone Number *", placeholder="+1-234-567-8900")
                priority = st.selectbox("Priority", ["normal", "high", "critical"])
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                submit = st.form_submit_button("📤 Submit")
            with col2:
                cancel = st.form_submit_button("❌ Cancel")
            
            if submit:
                if name and email and phone:
                    # Get the last user query
                    last_query = None
                    for msg in reversed(st.session_state.messages):
                        if msg['role'] == 'user':
                            last_query = msg['content']
                            break
                    
                    if last_query:
                        escalation_data = {
                            "name": name,
                            "email": email,
                            "phone": phone,
                            "priority": priority
                        }
                        
                        result = process_query(last_query, escalation_data)
                        
                        add_message(
                            'assistant',
                            result['response'],
                            {
                                'departments': result.get('departments_identified', []),
                                'sentiment': result.get('sentiment', ''),
                                'escalated': result.get('escalated', False)
                            }
                        )
                        
                        st.session_state.show_escalation_form = False
                        st.success("✅ Your information has been submitted!")
                        st.rerun()
                else:
                    st.error("Please fill in all required fields")
            
            if cancel:
                st.session_state.show_escalation_form = False
                st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Type your message here... (files will be included if uploaded)", disabled=not st.session_state.agent_initialized):
        if not st.session_state.agent_initialized:
            st.error("⚠️ Please initialize the agent first using the sidebar")
        else:
            # Process uploaded files
            processed_files = None
            if st.session_state.current_files:
                processed_files = process_uploaded_files(st.session_state.current_files)
                
                # Add file info to message
                file_names = [f.name for f in st.session_state.current_files]
                file_info = f"\n\n📎 Attached: {', '.join(file_names)}"
                prompt_with_files = prompt + file_info
            else:
                prompt_with_files = prompt
            
            # Add user message
            add_message('user', prompt_with_files)
            
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt_with_files)
            
            # Process query with files
            result = process_query(prompt, uploaded_files=processed_files)
            
            if isinstance(result, dict) and not result.get('error'):
                # Add assistant response
                response_content = result['response']
                
                # Add document analysis if present
                if result.get('document_analysis'):
                    response_content = f"**📄 Document Analysis:**\n\n{result['document_analysis']}\n\n---\n\n{response_content}"
                
                add_message(
                    'assistant',
                    response_content,
                    {
                        'departments': result.get('departments_identified', []),
                        'sentiment': result.get('sentiment', ''),
                        'escalated': result.get('escalated', False),
                        'has_files': processed_files is not None
                    }
                )
                
                # Check if escalation form should be shown
                if result.get('escalated') and 'Reference Number' not in result['response']:
                    st.session_state.show_escalation_form = True
            else:
                add_message('assistant', result if isinstance(result, str) else result['response'])
            
            # Clear uploaded files after processing
            st.session_state.current_files = []
            
            st.rerun()
    
    # Quick action buttons
    if st.session_state.agent_initialized and not st.session_state.show_escalation_form:
        st.markdown("---")
        st.markdown("### 💡 Quick Questions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        quick_questions = [
            "How do I return a product?",
            "What sales are currently running?",
            "How do I reset my password?",
            "Analyze my uploaded receipt"
        ]
        
        cols = [col1, col2, col3, col4]
        for i, question in enumerate(quick_questions):
            with cols[i]:
                if st.button(question, key=f"quick_{i}"):
                    # Add to messages and process
                    add_message('user', question)
                    result = process_query(question)
                    
                    if isinstance(result, dict) and not result.get('error'):
                        add_message(
                            'assistant',
                            result['response'],
                            {
                                'departments': result.get('departments_identified', []),
                                'sentiment': result.get('sentiment', ''),
                                'escalated': result.get('escalated', False)
                            }
                        )
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: white; padding: 20px;">
        <p>🛍️ ShopUNow AI Assistant v3.0 Multi-Modal | Powered by GPT-4 Vision & LangGraph</p>
        <p style="font-size: 0.8em;">Built with ❤️ for Analytics Vidya Capstone Project</p>
        <p style="font-size: 0.7em;">✨ Supports Text, Images, PDFs, and Documents</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()