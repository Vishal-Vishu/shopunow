import io
import base64
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

pytesseract.pytesseract.tesseract_cmd = r"D:\\Program Files\\Tesseract-OCR\\tesseract.exe"


load_dotenv()

vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)


def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_image_ocr(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)


def analyze_image_with_vision_llm(file):
    """
    Uses GPT-4o vision model for semantic understanding of image.
    Much more accurate than raw OCR for receipts.
    """
    image = Image.open(file)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    response = vision_llm.invoke([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all useful information from this receipt or bill."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                }
            ]
        }
    ])

    return response.content


def process_uploaded_file(uploaded_file):
    file_type = uploaded_file.type

    if "pdf" in file_type:
        return extract_text_from_pdf(uploaded_file)

    elif "word" in file_type or "docx" in file_type:
        return extract_text_from_docx(uploaded_file)

    elif "image" in file_type:
        # First try vision model (better)
        return analyze_image_with_vision_llm(uploaded_file)

    else:
        return "Unsupported file type."
import io
import base64
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document
from langchain_openai import ChatOpenAI

vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)


def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_image_ocr(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)


def analyze_image_with_vision_llm(file):
    """
    Uses GPT-4o vision model for semantic understanding of image.
    Much more accurate than raw OCR for receipts.
    """
    image = Image.open(file)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    response = vision_llm.invoke([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all useful information from this receipt or bill."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                }
            ]
        }
    ])

    return response.content


def process_uploaded_file(uploaded_file):
    file_type = uploaded_file.type

    if "pdf" in file_type:
        return extract_text_from_pdf(uploaded_file)

    elif "word" in file_type or "docx" in file_type:
        return extract_text_from_docx(uploaded_file)

    elif "image" in file_type:
        # First try vision model (better)
        return analyze_image_with_vision_llm(uploaded_file)

    else:
        return "Unsupported file type."
