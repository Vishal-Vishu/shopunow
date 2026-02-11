"""
ShopUNow Complete Multi-Modal Agentic AI Assistant
====================================================
Features:
- Dedicated LLM agents for each department with unique personas
- All departments (8 total: 4 external + 4 internal)
- Multi-modal support: Images, PDFs, Documents
- Vision capabilities for bills, receipts, invoices
- Multi-user conversational system with memory management
- Enhanced escalation with user details form
- FastAPI deployment ready

Author: Analytics Vidya Capstone Project
"""

import os
import json
import uuid
import base64
from datetime import datetime
from typing import TypedDict, List, Annotated, Dict, Optional
import operator
from io import BytesIO
from PIL import Image
import PyPDF2

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
# Removed: from langchain.memory import ConversationBufferMemory (deprecated)
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import department-specific agents
from department_agents import DepartmentAgentManager, DEPARTMENT_AGENT_CONFIGS

# ============================================================================
# CONFIGURATION - All 8 Departments (Same as before)
# ============================================================================

DEPARTMENTS_CONFIG = {
    "customer_service": {
        "name": "Customer Service",
        "type": "external",
        "description": "Handles customer inquiries about orders, returns, refunds, and general support"
    },
    "product_information": {
        "name": "Product Information",
        "type": "external",
        "description": "Provides information about products, availability, specifications, and recommendations"
    },
    "sales": {
        "name": "Sales & Promotions",
        "type": "external",
        "description": "Information about current sales, discounts, promotional offers, and deals"
    },
    "technical_support": {
        "name": "Technical Support",
        "type": "external",
        "description": "Troubleshooting and technical assistance for products purchased from ShopUNow"
    },
    "hr": {
        "name": "Human Resources",
        "type": "internal",
        "description": "Manages employee benefits, payroll, leave policies, and HR-related queries"
    },
    "it_support": {
        "name": "IT Support",
        "type": "internal",
        "description": "Handles technical issues, software access, system troubleshooting for employees"
    },
    "finance": {
        "name": "Finance & Accounting",
        "type": "internal",
        "description": "Employee expense reports, reimbursements, travel claims, and financial policies"
    },
    "operations": {
        "name": "Operations & Logistics",
        "type": "internal",
        "description": "Warehouse operations, inventory management, supply chain, and internal logistics"
    }
}

# Sample QA Data (keeping it concise for space - same 15 pairs per dept as before)
SAMPLE_QA_DATA = {
    "customer_service": [
        {"question": "How do I return a product?", "answer": "You can return any product within 30 days of purchase. Simply log into your account, go to 'My Orders', select the item you want to return, and click 'Return Item'. You'll receive a prepaid return label via email."},
        {"question": "What is your refund policy?", "answer": "We offer full refunds for items returned within 30 days in original condition with tags attached. Refunds are processed to the original payment method within 5-7 business days."},
        {"question": "How long does shipping take?", "answer": "Standard shipping takes 5-7 business days. Express shipping delivers in 2-3 business days. Free shipping is available on orders over $50."},
        {"question": "Can I change my order after placing it?", "answer": "You can modify or cancel your order within 1 hour of placing it. Go to 'My Orders' and click 'Edit Order'."},
        {"question": "What payment methods do you accept?", "answer": "We accept all major credit cards (Visa, MasterCard, American Express), debit cards, PayPal, Apple Pay, Google Pay, and ShopUNow gift cards."},
        {"question": "How do I track my order?", "answer": "Once your order ships, you'll receive an email with a tracking number. You can also track your order by logging into your account."},
        {"question": "Do you offer international shipping?", "answer": "Yes, we ship to over 50 countries. International shipping costs and delivery times vary by location."},
        {"question": "What if my item arrives damaged?", "answer": "If your item arrives damaged, please contact us within 48 hours with photos. We'll arrange for a replacement or full refund."},
        {"question": "Can I use multiple discount codes?", "answer": "Only one discount code can be applied per order. The system will automatically apply the code that gives you the best savings."},
        {"question": "How do I create an account?", "answer": "Click 'Sign Up' at the top right of our website. Enter your email, create a password, and fill in your basic information."},
        {"question": "What is your customer service hours?", "answer": "Our customer service team is available Monday-Friday 8AM-8PM EST, Saturday-Sunday 9AM-6PM EST."},
        {"question": "Can I pick up my order in store?", "answer": "Yes, we offer free in-store pickup at over 200 locations. Select 'Store Pickup' at checkout."},
        {"question": "How do I cancel my order?", "answer": "You can cancel your order within 1 hour of placing it through 'My Orders'. After that, contact customer service immediately."},
        {"question": "Do you offer gift wrapping?", "answer": "Yes, gift wrapping is available for $4.99 per item. Select the gift wrap option at checkout."},
        {"question": "How do I update my shipping address?", "answer": "You can update your shipping address in 'My Account' > 'Addresses'. For orders already placed, contact customer service within 1 hour."}
    ],
    "product_information": [
        {"question": "What brands do you carry?", "answer": "ShopUNow carries over 200 premium brands including Nike, Adidas, Apple, Samsung, Sony, KitchenAid, and exclusive in-house brands."},
        {"question": "Do you have a size guide?", "answer": "Yes, each clothing item has a detailed size guide on the product page. Click 'Size Guide' below the size selector."},
        {"question": "Are products covered by warranty?", "answer": "All electronics come with manufacturer's warranty (typically 1 year). We also offer optional extended warranty plans."},
        {"question": "How do I know if an item is in stock?", "answer": "Stock availability is shown on each product page. If an item is out of stock, you can click 'Notify Me' to receive an email."},
        {"question": "Can I see product reviews?", "answer": "Yes, scroll down to the 'Customer Reviews' section on any product page. You can filter reviews by star rating and verified purchases."},
        {"question": "Do you offer price matching?", "answer": "Yes, we match prices from major authorized retailers. Submit a price match request within 7 days of purchase."},
        {"question": "What is your exchange policy?", "answer": "We offer free exchanges within 30 days. Select 'Exchange' in your order history and choose the new size/color."},
        {"question": "Are refurbished products available?", "answer": "Yes, we offer certified refurbished electronics at discounted prices. All refurbished items come with a 90-day warranty."},
        {"question": "Can I pre-order upcoming products?", "answer": "Yes, pre-orders are available for select products. You'll see a 'Pre-Order' button instead of 'Add to Cart'."},
        {"question": "Do you have product comparison tools?", "answer": "Yes, click 'Compare' on product pages to add up to 4 items for side-by-side comparison."},
        {"question": "What's your newest product line?", "answer": "We just launched the ShopUNow Home Smart collection featuring IoT-enabled appliances and smart lighting."},
        {"question": "Do you sell gift cards?", "answer": "Yes, ShopUNow gift cards are available in denominations from $25 to $500."},
        {"question": "Can I get product recommendations?", "answer": "Yes, our AI-powered recommendation engine suggests products based on your browsing and purchase history."},
        {"question": "What are your most popular products?", "answer": "Our bestsellers include Apple AirPods Pro, Samsung QLED TVs, Nike Air Max sneakers, and KitchenAid stand mixers."},
        {"question": "Do you offer product bundles?", "answer": "Yes, we offer bundled deals on complementary products at discounted prices. Look for 'Bundle & Save' tags."}
    ],
    "sales": [
        {"question": "What sales are currently running?", "answer": "We're currently running our Spring Clearance Sale with up to 60% off select items. Plus, new members get 20% off their first purchase."},
        {"question": "When is your next big sale event?", "answer": "Our next major sale is the Summer Mega Sale starting July 1st with deals across all categories."},
        {"question": "How do I use a promo code?", "answer": "Enter your promo code in the 'Promo Code' box at checkout before completing your purchase."},
        {"question": "Do you have a loyalty program?", "answer": "Yes! ShopUNow Rewards gives you 1 point per dollar spent. Earn 100 points and get a $5 reward."},
        {"question": "Are sale items returnable?", "answer": "Yes, most sale items are returnable within 30 days. However, items marked 'Final Sale' cannot be returned."},
        {"question": "How often do you restock sale items?", "answer": "Popular sale items may restock during the promotion if available, but it's not guaranteed."},
        {"question": "Can I combine sale prices with coupons?", "answer": "Most sales can be combined with percentage-off coupons, but not with dollar-amount coupons."},
        {"question": "Do you offer student discounts?", "answer": "Yes, verified students get 15% off year-round through our Student Discount program."},
        {"question": "Is there a military discount?", "answer": "Absolutely! We offer 10% off to active military, veterans, and their families."},
        {"question": "What's your Black Friday sale like?", "answer": "Our Black Friday sale is our biggest event with deals up to 70% off. Early access for ShopUNow Rewards members."},
        {"question": "Do you have a price adjustment policy?", "answer": "Yes, if an item you purchased goes on sale within 14 days, we'll refund the difference."},
        {"question": "Are there senior citizen discounts?", "answer": "Yes, customers 65+ receive 10% off every Tuesday in-store and online."},
        {"question": "How do I get notified about upcoming sales?", "answer": "Sign up for our email newsletter and text alerts to get notified about upcoming sales and exclusive offers."},
        {"question": "What's the employee referral discount?", "answer": "If you're referred by a ShopUNow employee, you get 25% off your first purchase."},
        {"question": "Do you have clearance sections?", "answer": "Yes, visit our Clearance Center online or in-store for items up to 80% off."}
    ],
    "technical_support": [
        {"question": "My new laptop won't turn on", "answer": "First, ensure the battery is charged by connecting the power adapter for at least 30 minutes. Try a hard reset by holding the power button for 15 seconds."},
        {"question": "How do I set up my new smart TV?", "answer": "Unbox and place the TV, connect the power cable, and turn it on. Follow the on-screen setup wizard to connect to WiFi."},
        {"question": "My wireless headphones won't pair", "answer": "Put the headphones in pairing mode (usually holding the power button for 5-7 seconds until LED flashes). On your device, go to Bluetooth settings."},
        {"question": "Can you help me install software?", "answer": "Yes! We offer free technical support for software installation within 30 days of purchase."},
        {"question": "My phone screen is cracked", "answer": "Screen damage is typically not covered by manufacturer warranty. However, if you purchased our ShopUNow Protection Plan, accidental damage is covered."},
        {"question": "How do I connect my printer to WiFi?", "answer": "Access the printer's control panel, go to Network settings, select WiFi Setup Wizard, choose your network, and enter the password."},
        {"question": "My smart watch isn't syncing", "answer": "Ensure Bluetooth is enabled on both devices. Restart both devices. In the companion app, try disconnecting and reconnecting."},
        {"question": "Can I get help transferring data?", "answer": "Absolutely! We offer free data transfer assistance for phones purchased at ShopUNow."},
        {"question": "My appliance is making strange noises", "answer": "Check if the appliance is level and on a stable surface. Ensure nothing is caught in moving parts. If noise persists, contact support."},
        {"question": "How do I update firmware?", "answer": "Most devices update automatically when connected to WiFi. To manually check: go to Settings > About > Software Update."},
        {"question": "My camera lens is foggy inside", "answer": "Internal lens fogging may indicate moisture damage. This is typically covered under warranty. Do not attempt to open the camera."},
        {"question": "Can you recommend antivirus software?", "answer": "For computers purchased at ShopUNow, we recommend Norton 360 or McAfee Total Protection at discounted rates."},
        {"question": "My smart home devices aren't responding", "answer": "Check that your voice assistant device is connected to WiFi. Try saying 'discover devices' to your assistant."},
        {"question": "How do I calibrate my monitor?", "answer": "Most monitors have preset color profiles. For basic calibration, use your computer's built-in calibration tool."},
        {"question": "My device keeps overheating", "answer": "Ensure vents aren't blocked, close unused apps, and use in a cool environment. If overheating continues, it may require service."}
    ],
    "hr": [
        {"question": "How do I request time off?", "answer": "Log into the employee portal and navigate to 'Time Off Requests'. Select the dates and type of leave. Submit at least 2 weeks in advance."},
        {"question": "What are the health insurance options?", "answer": "ShopUNow offers three plans: Basic (HMO), Standard (PPO), and Premium (PPO Plus). Open enrollment is in November."},
        {"question": "When is payday?", "answer": "Employees are paid bi-weekly on Fridays. Paystubs can be accessed through the employee portal."},
        {"question": "How do I update my personal information?", "answer": "Log into the employee portal and go to 'My Profile'. You can update address, phone number, and emergency contacts."},
        {"question": "What is the 401(k) match policy?", "answer": "ShopUNow matches 100% of employee contributions up to 4% of salary. You're eligible after 90 days."},
        {"question": "How many vacation days do I get?", "answer": "New employees receive 15 days PTO annually. After 3 years, this increases to 20 days. After 5 years, 25 days."},
        {"question": "What is the dress code policy?", "answer": "Business casual Monday-Thursday. Casual Friday allows jeans and sneakers. Warehouse employees wear provided uniforms."},
        {"question": "How do I access my W-2 form?", "answer": "W-2 forms are available by January 31st each year in the employee portal under 'Tax Documents'."},
        {"question": "What employee discounts are available?", "answer": "All employees receive 30% off ShopUNow products year-round. Additional discounts during employee appreciation weeks."},
        {"question": "What is the parental leave policy?", "answer": "Birth mothers receive 12 weeks paid leave. Non-birth parents receive 6 weeks paid leave."},
        {"question": "How do I report harassment?", "answer": "Report immediately to HR via confidential hotline at ext. 7777 or email ethics@shopunow.com."},
        {"question": "What are career advancement opportunities?", "answer": "ShopUNow promotes from within whenever possible. We also offer tuition reimbursement up to $5,000/year."},
        {"question": "Can I work remotely?", "answer": "Remote work eligibility depends on your role. Corporate positions may qualify for hybrid (2-3 days in office)."},
        {"question": "What if I'm sick and out of PTO?", "answer": "After exhausting PTO, you may use unpaid sick leave. For serious conditions, you may qualify for FMLA."},
        {"question": "How do I enroll in stock purchase plan?", "answer": "Eligible employees can purchase ShopUNow stock at 15% discount through payroll deductions."}
    ],
    "it_support": [
        {"question": "How do I reset my password?", "answer": "Go to the login page and click 'Forgot Password'. Enter your company email and you'll receive a reset link."},
        {"question": "Who do I contact for laptop issues?", "answer": "Submit a ticket through the IT Support portal or email itsupport@shopunow.com. For urgent issues, call ext. 4357."},
        {"question": "How do I request software access?", "answer": "Submit a Software Access Request through the IT portal. Include software name, business justification, and manager approval."},
        {"question": "What is the VPN setup process?", "answer": "Download Cisco AnyConnect from the IT portal. Install it and use your company credentials to connect."},
        {"question": "How do I connect to office WiFi?", "answer": "Connect to 'ShopUNow-Corporate' network. Enter your company email and password."},
        {"question": "What are email storage limits?", "answer": "Email mailboxes have a 50GB limit. Archive old emails to free up space."},
        {"question": "How do I set up MFA?", "answer": "Download Microsoft Authenticator app. Go to account.shopunow.com, click 'Security', then 'Enable MFA'."},
        {"question": "Can I use personal devices for work?", "answer": "Personal devices must be registered through the BYOD program. Install the company security profile."},
        {"question": "What is the policy on software installations?", "answer": "Employees cannot install unauthorized software. All installations must go through IT."},
        {"question": "How do I report a security incident?", "answer": "Immediately email security@shopunow.com or call the security hotline at ext. 9999."},
        {"question": "My computer is running slow", "answer": "Close unnecessary applications and browser tabs. Restart your computer. Clear browser cache."},
        {"question": "How do I access shared drives?", "answer": "Shared drives are accessible through File Explorer under Network Locations."},
        {"question": "Can I install browser extensions?", "answer": "Browser extensions require IT approval. Submit requests through the IT portal."},
        {"question": "How do I set up work email on phone?", "answer": "Download Microsoft Outlook app. Enter your work email when prompted."},
        {"question": "What if I forgot to lock my computer?", "answer": "Computers auto-lock after 5 minutes. Repeated violations may require security training."}
    ],
    "finance": [
        {"question": "How do I submit an expense report?", "answer": "Log into Concur through the employee portal. Click 'New Expense Report', add expenses with receipts. Reimbursements processed within 10 business days."},
        {"question": "What expenses are reimbursable?", "answer": "Reimbursable expenses include business travel, client meals, conference fees, parking, and approved supplies."},
        {"question": "How long to submit expenses?", "answer": "Expense reports must be submitted within 30 days of the expense date."},
        {"question": "What is the per diem rate?", "answer": "Domestic per diem is $65/day for meals and incidentals. International rates vary by location."},
        {"question": "How do I get a corporate credit card?", "answer": "Employees who travel regularly can request a corporate card through the Finance portal with manager approval."},
        {"question": "What is mileage reimbursement rate?", "answer": "Current IRS mileage rate is $0.655 per mile for business use of personal vehicle."},
        {"question": "How do I request travel advance?", "answer": "For trips exceeding $1,000, request through Finance portal at least 2 weeks before travel."},
        {"question": "What's policy on first-class travel?", "answer": "First-class requires VP approval and is only approved for international flights over 6 hours."},
        {"question": "How do I split expenses?", "answer": "One person submits full receipt and notes 'split with [names]'. Each person submits their portion."},
        {"question": "Can I expense client gifts?", "answer": "Yes, client gifts up to $50 per person per year are allowed and must be business-appropriate."},
        {"question": "What documentation for international travel?", "answer": "Provide receipts in original currency along with conversion to USD. Keep all receipts and boarding passes."},
        {"question": "How are taxes handled on reimbursements?", "answer": "Properly documented business expense reimbursements are not taxable income."},
        {"question": "What if I lost a receipt?", "answer": "For lost receipts under $25, submit with a lost receipt affidavit. Over $25 requires vendor documentation."},
        {"question": "How do I reconcile corporate card?", "answer": "Review your corporate card statement in Concur monthly. Match each charge to an expense report."},
        {"question": "What's policy on conference attendance?", "answer": "Conference attendance requires manager approval in advance with conference details and business justification."}
    ],
    "operations": [
        {"question": "How do I request inventory?", "answer": "Submit an inventory request through the Operations portal. Include item description, quantity, and business justification."},
        {"question": "How to report damaged inventory?", "answer": "Report immediately through the Warehouse Management System. Document with photos and complete damage report form."},
        {"question": "How do I track a shipment?", "answer": "All shipments can be tracked in the WMS using the shipment ID. You'll receive automated updates at key milestones."},
        {"question": "What are warehouse hours?", "answer": "Main warehouse operates Monday-Friday 6AM-10PM, Saturday 7AM-6PM, closed Sundays."},
        {"question": "How do I schedule pickup/delivery?", "answer": "Schedule through Operations portal at least 48 hours in advance. Include pickup location, delivery destination, and item details."},
        {"question": "What safety protocols in warehouse?", "answer": "All personnel must wear safety vests, steel-toed boots, and hard hats in designated zones."},
        {"question": "How is inventory counted?", "answer": "Cycle counts occur daily for high-velocity items. Full physical inventory conducted quarterly."},
        {"question": "What's the RMA process?", "answer": "Customer returns processed through RMA system. Inspect items, update condition codes, and route to disposition."},
        {"question": "How do I report equipment malfunction?", "answer": "Report immediately through maintenance portal or call facilities at ext. 5600."},
        {"question": "What's procedure for hazardous materials?", "answer": "Hazmat must be stored in designated areas with proper labeling. Only certified personnel can handle."},
        {"question": "How do I access supply chain dashboard?", "answer": "Available in Operations portal under 'Analytics'. View real-time inventory levels and order status."},
        {"question": "What's policy on overtime?", "answer": "Overtime must be pre-approved by supervisor. During peak seasons, extended hours expected."},
        {"question": "How do I onboard new warehouse staff?", "answer": "New employees complete safety orientation, equipment certification, and WMS training in first week."},
        {"question": "What are quality control standards?", "answer": "All outbound shipments undergo QC inspection. Random sampling at 10% for standard items."},
        {"question": "How do I request equipment maintenance?", "answer": "Submit maintenance request through portal with equipment ID and issue description."}
    ]
}

# ============================================================================
# ENHANCED STATE DEFINITIONS - WITH MULTI-MODAL SUPPORT
# ============================================================================

class AgentState(TypedDict):
    user_id: str
    session_id: str
    query: str
    sentiment: str
    departments: List[str]
    route: str
    responses: Annotated[List[str], operator.add]
    final_response: str
    escalate: bool
    escalation_details: Optional[Dict]
    conversation_history: List[Dict]
    # New: Multi-modal fields
    uploaded_files: Optional[List[Dict]]  # Contains file data
    document_analysis: Optional[str]  # Analysis of uploaded documents
    has_visual_content: bool

class EscalationForm(BaseModel):
    name: str
    email: str
    phone: str
    priority: str = "normal"

class QueryRequest(BaseModel):
    user_id: str
    query: str
    session_id: Optional[str] = None
    escalation_form: Optional[EscalationForm] = None

class QueryResponse(BaseModel):
    response: str
    session_id: str
    departments_identified: List[str]
    sentiment: str
    escalated: bool
    document_analysis: Optional[str] = None

# ============================================================================
# MULTI-MODAL DOCUMENT PROCESSING UTILITIES
# ============================================================================

class DocumentProcessor:
    """Handles processing of images, PDFs, and documents"""
    
    @staticmethod
    def encode_image_to_base64(image_data) -> str:
        """Convert image to base64 string"""
        if isinstance(image_data, bytes):
            return base64.b64encode(image_data).decode('utf-8')
        elif isinstance(image_data, Image.Image):
            buffered = BytesIO()
            image_data.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        return ""
    
    @staticmethod
    def process_pdf(pdf_data: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_data))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    @staticmethod
    def get_file_type(filename: str) -> str:
        """Determine file type from filename"""
        ext = filename.lower().split('.')[-1]
        if ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
            return 'image'
        elif ext == 'pdf':
            return 'pdf'
        elif ext in ['txt', 'doc', 'docx']:
            return 'document'
        return 'unknown'

# ============================================================================
# ENHANCED AGENT WITH MULTI-MODAL SUPPORT
# ============================================================================

class ShopUNowEnhancedAgent:
    def __init__(self, openai_api_key: str, enable_email: bool = False):
        # Use GPT-4 Vision for multi-modal support (coordinator)
        self.llm = ChatOpenAI(
            model="gpt-4o",  # Supports vision
            temperature=0,
            api_key=openai_api_key
        )
        self.llm_text_only = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.vector_store = None
        self.departments = list(DEPARTMENTS_CONFIG.keys())
        self.enable_email = enable_email
        self.doc_processor = DocumentProcessor()
        
        # Initialize department-specific agents
        self.dept_agents = DepartmentAgentManager(openai_api_key)
        print("✓ Department-specific agents initialized with unique personas")
        
        # Multi-user memory management (using dict instead of LangChain memory)
        self.session_histories: Dict[str, List[Dict]] = {}
        
    def get_or_create_session_history(self, user_id: str, session_id: str) -> List[Dict]:
        """Get or create conversation history for a user session"""
        key = f"{user_id}_{session_id}"
        if key not in self.session_histories:
            self.session_histories[key] = []
        return self.session_histories[key]
    
    def add_to_history(self, user_id: str, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        key = f"{user_id}_{session_id}"
        if key not in self.session_histories:
            self.session_histories[key] = []
        
        self.session_histories[key].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
    def get_conversation_context(self, user_id: str, session_id: str) -> str:
        """Get conversation history as context"""
        key = f"{user_id}_{session_id}"
        history = self.session_histories.get(key, [])
        
        if not history:
            return ""
        
        context = "Previous conversation:\n"
        for msg in history[-5:]:
            context += f"{msg['role']}: {msg['content']}\n"
        return context
        
    def create_vector_database(self):
        """Create vector database with all 8 departments"""
        documents = []
        
        for dept_id, qa_pairs in SAMPLE_QA_DATA.items():
            dept_info = DEPARTMENTS_CONFIG[dept_id]
            for qa in qa_pairs:
                content = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
                doc = Document(
                    page_content=content,
                    metadata={
                        "department": dept_id,
                        "department_name": dept_info["name"],
                        "type": dept_info["type"],
                        "question": qa["question"],
                        "answer": qa["answer"]
                    }
                )
                documents.append(doc)
        
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        print(f"✓ Vector database created with {len(documents)} documents across {len(DEPARTMENTS_CONFIG)} departments")
    
    # ========================================================================
    # NEW: MULTI-MODAL DOCUMENT ANALYSIS
    # ========================================================================
    
    def analyze_document(self, state: AgentState) -> AgentState:
        """Analyze uploaded documents using vision capabilities"""
        if not state.get("uploaded_files") or len(state["uploaded_files"]) == 0:
            state["document_analysis"] = None
            state["has_visual_content"] = False
            return state
        
        state["has_visual_content"] = True
        analyses = []
        
        for file_info in state["uploaded_files"]:
            file_type = file_info.get("type")
            file_data = file_info.get("data")
            filename = file_info.get("filename", "unknown")
            
            try:
                if file_type == "image":
                    # Use GPT-4 Vision to analyze image
                    base64_image = self.doc_processor.encode_image_to_base64(file_data)
                    
                    vision_prompt = f"""Analyze this image uploaded by a ShopUNow customer or employee.
                    
This could be:
- A receipt or invoice (extract order details, items, prices, dates)
- A product image (identify product, condition, any damage)
- An expense report (extract amounts, dates, categories)
- A bill or statement (identify key information)
- An ID or document (extract relevant details while respecting privacy)
- A damaged product photo (describe the damage)

Provide a detailed analysis including:
1. Document type
2. Key information extracted
3. Any issues or concerns visible
4. Recommended department for handling this query

User's question: {state['query']}

Be thorough and extract all relevant information."""

                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": vision_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    )
                    
                    response = self.llm.invoke([message])
                    analysis = f"**Analysis of {filename}:**\n{response.content}"
                    analyses.append(analysis)
                    print(f"✓ Analyzed image: {filename}")
                    
                elif file_type == "pdf":
                    # Extract text from PDF
                    pdf_text = self.doc_processor.process_pdf(file_data)
                    
                    if pdf_text and len(pdf_text) > 50:
                        # Analyze the extracted text
                        pdf_prompt = f"""Analyze this PDF document content from a ShopUNow customer or employee.

Document content:
{pdf_text[:3000]}  # First 3000 chars

User's question: {state['query']}

Provide analysis including:
1. Document type
2. Key information (dates, amounts, items, people)
3. Relevant details for the user's question
4. Recommended department for handling"""

                        response = self.llm_text_only.invoke(pdf_prompt)
                        analysis = f"**Analysis of {filename}:**\n{response.content}"
                        analyses.append(analysis)
                        print(f"✓ Analyzed PDF: {filename}")
                    else:
                        analyses.append(f"**{filename}:** Could not extract text from PDF.")
                        
            except Exception as e:
                analyses.append(f"**{filename}:** Error analyzing - {str(e)}")
                print(f"❌ Error analyzing {filename}: {e}")
        
        state["document_analysis"] = "\n\n".join(analyses) if analyses else None
        return state
    
    # ========================================================================
    # ENHANCED NODE FUNCTIONS WITH MULTI-MODAL AWARENESS
    # ========================================================================
    
    def analyze_query(self, state: AgentState) -> AgentState:
        """Analyze sentiment and identify departments with document context"""
        query = state["query"]
        conversation_context = self.get_conversation_context(
            state["user_id"], 
            state["session_id"]
        )
        
        # Include document analysis in context
        doc_context = ""
        if state.get("document_analysis"):
            doc_context = f"\n\nUploaded document analysis:\n{state['document_analysis']}\n"
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query analyzer for ShopUNow retail company.

Available departments:
{departments}

{context}
{doc_context}

Analyze the user query and provide:
1. Sentiment: positive, neutral, or negative
2. Relevant departments (can be multiple if query spans departments)

If documents are uploaded, consider their content when identifying departments.

Return ONLY a JSON object:
{{"sentiment": "positive|neutral|negative", "departments": ["dept1", "dept2"]}}

Examples:
- Receipt image + "Why was I charged twice?" -> {{"sentiment": "negative", "departments": ["customer_service", "finance"]}}
- Expense report PDF + "Is this format correct?" -> {{"sentiment": "neutral", "departments": ["finance"]}}
- Product damage photo + "Can I return this?" -> {{"sentiment": "neutral", "departments": ["customer_service"]}}
"""),
            ("user", "{query}")
        ])
        
        dept_list = "\n".join([f"- {k}: {v['description']}" for k, v in DEPARTMENTS_CONFIG.items()])
        context_text = f"\n{conversation_context}" if conversation_context else ""
        
        response = self.llm_text_only.invoke(
            analysis_prompt.format_messages(
                query=query, 
                departments=dept_list,
                context=context_text,
                doc_context=doc_context
            )
        )
        
        try:
            analysis = json.loads(response.content)
            state["sentiment"] = analysis["sentiment"]
            state["departments"] = analysis["departments"]
        except:
            state["sentiment"] = "neutral"
            state["departments"] = ["customer_service"]
        
        print(f"📊 Analysis - Sentiment: {state['sentiment']}, Departments: {state['departments']}")
        return state
    
    def route_decision(self, state: AgentState) -> AgentState:
        """Enhanced routing with custom workflows"""
        sentiment = state["sentiment"]
        departments = state["departments"]
        query = state["query"].lower()
        
        valid_depts = [d for d in departments if d in self.departments]
        
        urgent_keywords = ['urgent', 'emergency', 'asap', 'immediately', 'critical']
        is_urgent = any(keyword in query for keyword in urgent_keywords)
        
        if sentiment == "negative" or len(valid_depts) == 0 or is_urgent:
            state["route"] = "escalate"
            state["escalate"] = True
        else:
            state["route"] = "rag"
            state["escalate"] = False
        
        print(f"🔀 Route Decision: {state['route']}")
        return state
    
    def handle_escalation(self, state: AgentState) -> AgentState:
        """Enhanced escalation with form handling"""
        escalation_details = state.get("escalation_details")
        
        if escalation_details:
            name = escalation_details.get("name")
            email = escalation_details.get("email")
            phone = escalation_details.get("phone")
            priority = escalation_details.get("priority", "normal")
            
            if self.enable_email:
                self.send_escalation_email(name, email, phone, state["query"], priority)
            
            doc_notice = ""
            if state.get("has_visual_content"):
                doc_notice = "\n\n📎 **Attached documents have been forwarded to our support team for review.**"
            
            escalation_msg = f"""Thank you, {name}! 

Your concern has been received and forwarded to our priority support team.

**Your Details:**
- Name: {name}
- Email: {email}
- Phone: {phone}
- Priority: {priority.upper()}

**Reference Number:** SN{uuid.uuid4().hex[:8].upper()}

A specialist will contact you at {phone} or {email} within:
- Critical: 2 hours
- High: 4 hours  
- Normal: 24 hours

{doc_notice}

We appreciate your patience and will resolve this promptly."""
        else:
            doc_notice = ""
            if state.get("has_visual_content"):
                doc_notice = "\n\n📎 We've received your uploaded documents and they will be reviewed by our support team."
            
            escalation_msg = f"""Thank you for contacting ShopUNow.

I understand your concern requires special attention. To expedite your request, please provide:

1. Your full name
2. Contact email
3. Phone number
4. Brief description of your issue

You can also call our priority line at 1-800-SHOPUNOW for immediate assistance.

**Reference Number:** SN{uuid.uuid4().hex[:8].upper()}

{doc_notice}

We're committed to resolving this quickly!"""
        
        state["final_response"] = escalation_msg
        state["responses"] = [state["final_response"]]
        
        self.add_to_history(
            state["user_id"], 
            state["session_id"], 
            "assistant", 
            "Escalated to human support"
        )
        
        print("📞 Escalated to human agent")
        return state
    
    def send_escalation_email(self, name: str, email: str, phone: str, query: str, priority: str):
        """Send email notification for escalation"""
        try:
            smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            sender_email = os.getenv("SENDER_EMAIL")
            sender_password = os.getenv("SENDER_PASSWORD")
            support_email = os.getenv("SUPPORT_EMAIL", "support@shopunow.com")
            
            if not sender_email or not sender_password:
                print("⚠️  Email credentials not configured")
                return
            
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = support_email
            msg['Subject'] = f"[{priority.upper()}] Customer Escalation - {name}"
            
            body = f"""
New customer escalation received:

Priority: {priority.upper()}
Name: {name}
Email: {email}
Phone: {phone}

Query:
{query}

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            
            print(f"✉️  Escalation email sent to {support_email}")
        except Exception as e:
            print(f"❌ Email send failed: {e}")
    
    def rag_response(self, state: AgentState) -> AgentState:
        """Generate RAG-based responses using department-specific agents"""
        query = state["query"]
        departments = state["departments"]
        conversation_context = self.get_conversation_context(
            state["user_id"], 
            state["session_id"]
        )
        
        # Include document analysis
        doc_context = ""
        if state.get("document_analysis"):
            doc_context = state["document_analysis"]
        
        responses = []
        
        for dept in departments:
            if dept not in self.departments:
                continue
            
            # Get department-specific agent
            dept_agent = self.dept_agents.get_agent(dept)
            if not dept_agent:
                print(f"⚠️  No agent found for {dept}, skipping")
                continue
            
            # Retrieve relevant documents from vector store
            # FIXED: Use invoke() instead of get_relevant_documents()
            retriever = self.vector_store.as_retriever(
                search_kwargs={
                    "k": 3,
                    "filter": {"department": dept}
                }
            )
            
            # Use invoke() method for newer LangChain versions
            try:
                docs = retriever.invoke(query)
            except AttributeError:
                # Fallback for older versions
                docs = retriever.get_relevant_documents(query)
            
            # Build knowledge base context
            knowledge_context = "\n\n".join([
                f"Q: {doc.metadata['question']}\nA: {doc.metadata['answer']}"
                for doc in docs
            ])
            
            # Get persona info for response header
            persona_info = dept_agent.get_persona_info()
            
            # Generate response using department-specific agent
            agent_response = dept_agent.generate_response(
                query=query,
                context=knowledge_context,
                conversation_history=conversation_context,
                document_analysis=doc_context
            )
            
            # Format with persona header
            dept_response = f"**{DEPARTMENTS_CONFIG[dept]['name']}** (with {persona_info['name']}, {persona_info['role']}):\n\n{agent_response}"
            responses.append(dept_response)
            
            print(f"✓ Generated response from {persona_info['name']} ({dept})")
        
        state["responses"] = responses
        
        self.add_to_history(
            state["user_id"], 
            state["session_id"], 
            "user", 
            query
        )
        
        return state
    
    def compile_response(self, state: AgentState) -> AgentState:
        """Compile final response"""
        if state.get("escalate"):
            return state
            
        responses = state["responses"]
        
        # Add document analysis summary if present
        doc_summary = ""
        if state.get("document_analysis") and state.get("has_visual_content"):
            doc_summary = "\n\n---\n\n**📎 Document Analysis:**\n" + state["document_analysis"] + "\n\n---\n\n"
        
        if len(responses) == 1:
            state["final_response"] = doc_summary + responses[0]
        else:
            header = "I'll address each part of your question:\n\n"
            compiled = doc_summary + header + "\n\n---\n\n".join(responses)
            state["final_response"] = compiled
        
        self.add_to_history(
            state["user_id"], 
            state["session_id"], 
            "assistant", 
            state["final_response"]
        )
        
        return state
    
    # ========================================================================
    # BUILD LANGGRAPH WITH DOCUMENT ANALYSIS
    # ========================================================================
    
    def build_graph(self):
        """Build the LangGraph workflow with document analysis"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("analyze_docs", self.analyze_document)
        workflow.add_node("analyze", self.analyze_query)
        workflow.add_node("route", self.route_decision)
        workflow.add_node("escalate", self.handle_escalation)
        workflow.add_node("rag", self.rag_response)
        workflow.add_node("compile", self.compile_response)
        
        workflow.set_entry_point("analyze_docs")
        workflow.add_edge("analyze_docs", "analyze")
        workflow.add_edge("analyze", "route")
        
        workflow.add_conditional_edges(
            "route",
            lambda x: x["route"],
            {
                "escalate": "escalate",
                "rag": "rag"
            }
        )
        
        workflow.add_edge("rag", "compile")
        workflow.add_edge("escalate", END)
        workflow.add_edge("compile", END)
        
        return workflow.compile()
    
    def query(self, user_id: str, user_query: str, session_id: str = None, 
              escalation_form: Dict = None, uploaded_files: List[Dict] = None) -> Dict:
        """Process a user query with optional file uploads"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        initial_state = {
            "user_id": user_id,
            "session_id": session_id,
            "query": user_query,
            "sentiment": "",
            "departments": [],
            "route": "",
            "responses": [],
            "final_response": "",
            "escalate": False,
            "escalation_details": escalation_form,
            "conversation_history": [],
            "uploaded_files": uploaded_files or [],
            "document_analysis": None,
            "has_visual_content": False
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "response": result["final_response"],
            "session_id": session_id,
            "departments_identified": result["departments"],
            "sentiment": result["sentiment"],
            "escalated": result["escalate"],
            "document_analysis": result.get("document_analysis")
        }

# ============================================================================
# FASTAPI APPLICATION WITH FILE UPLOAD SUPPORT
# ============================================================================

app = FastAPI(title="ShopUNow Multi-Modal AI Assistant API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_instance = None

@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global agent_instance
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set")
        return
    
    print("🚀 Initializing ShopUNow Multi-Modal AI Assistant...")
    agent_instance = ShopUNowEnhancedAgent(api_key, enable_email=False)
    agent_instance.create_vector_database()
    agent_instance.graph = agent_instance.build_graph()
    print("✓ API Ready with Multi-Modal Support!")

@app.get("/")
async def root():
    return {
        "message": "ShopUNow Multi-Modal AI Assistant API",
        "version": "3.0",
        "features": ["text", "images", "pdfs", "documents"],
        "departments": list(DEPARTMENTS_CONFIG.keys())
    }

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process user query through the agent"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    escalation_form_dict = None
    if request.escalation_form:
        escalation_form_dict = {
            "name": request.escalation_form.name,
            "email": request.escalation_form.email,
            "phone": request.escalation_form.phone,
            "priority": request.escalation_form.priority
        }
    
    result = agent_instance.query(
        user_id=request.user_id,
        user_query=request.query,
        session_id=request.session_id,
        escalation_form=escalation_form_dict
    )
    
    return QueryResponse(**result)

@app.post("/api/query-with-files")
async def process_query_with_files(
    user_id: str = Form(...),
    query: str = Form(...),
    session_id: str = Form(None),
    files: List[UploadFile] = File(None)
):
    """Process query with file uploads"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    uploaded_files = []
    if files:
        for file in files:
            file_data = await file.read()
            file_type = DocumentProcessor.get_file_type(file.filename)
            
            uploaded_files.append({
                "filename": file.filename,
                "type": file_type,
                "data": file_data,
                "size": len(file_data)
            })
    
    result = agent_instance.query(
        user_id=user_id,
        user_query=query,
        session_id=session_id,
        uploaded_files=uploaded_files
    )
    
    return result

# ============================================================================
# MAIN EXECUTION & TESTING
# ============================================================================

def main():
    """Test the complete multi-modal system"""
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    print("=" * 100)
    print("🚀 SHOPUNOW MULTI-MODAL AGENTIC AI SYSTEM")
    print("=" * 100)
    print(f"\n📦 Departments: {len(DEPARTMENTS_CONFIG)}")
    print("🎨 Multi-Modal Support: Images, PDFs, Documents")
    print()
    
    agent = ShopUNowEnhancedAgent(api_key, enable_email=False)
    agent.create_vector_database()
    agent.graph = agent.build_graph()
    
    print("\n" + "=" * 100)
    print("TESTING: Text Queries")
    print("=" * 100)
    
    test_queries = [
        "How do I return a product?",
        "What sales do you have on laptops and what's the warranty?",
        "How do I submit expenses and request PTO?",
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"{'=' * 80}\n")
        
        result = agent.query("user_test", query)
        print(f"Response: {result['response'][:200]}...")
        print(f"Departments: {result['departments_identified']}")
        print("-" * 80)
    
    print("\n" + "=" * 100)
    print("✅ MULTI-MODAL SYSTEM READY!")
    print("=" * 100)
    print("\n📝 Features Implemented:")
    print("   ✅ Image analysis (receipts, bills, damage photos)")
    print("   ✅ PDF processing (invoices, expense reports)")
    print("   ✅ Document extraction and analysis")
    print("   ✅ Vision-powered GPT-4 integration")
    print("   ✅ Context-aware responses with document data")
    print("   ✅ All 8 departments + multi-user + memory")
    
    print("\n🎯 Use Cases:")
    print("   📸 Upload receipt → Ask about charges")
    print("   📄 Upload expense report → Ask if format is correct")
    print("   🖼️ Upload damage photo → Request return")
    print("   📋 Upload invoice → Query billing details")

if __name__ == "__main__":
    main()