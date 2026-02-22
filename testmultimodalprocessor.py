import os
from multimodalprocessor import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_image_ocr,
    analyze_image_with_vision_llm,
    process_uploaded_file
)

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TEST_FILES_DIR = "test_files"  # create this folder and add sample files


def test_pdf():
    print("\n📄 Testing PDF Extraction...")
    file_path = os.path.join(TEST_FILES_DIR, "sample.pdf")

    with open(file_path, "rb") as f:
        text = extract_text_from_pdf(f)

    print("Extracted PDF Text:\n")
    print(text[:500])


def test_docx():
    print("\n📝 Testing DOCX Extraction...")
    file_path = os.path.join(TEST_FILES_DIR, "sample.docx")

    text = extract_text_from_docx(file_path)

    print("Extracted DOCX Text:\n")
    print(text[:500])


def test_image_ocr():
    print("\n🖼 Testing Image OCR...")
    file_path = os.path.join(TEST_FILES_DIR, "sample.png")

    with open(file_path, "rb") as f:
        text = extract_text_from_image_ocr(f)

    print("OCR Extracted Text:\n")
    print(text)


def test_image_vision():
    print("\n🧠 Testing Vision LLM...")
    file_path = os.path.join(TEST_FILES_DIR, "sample.png")

    with open(file_path, "rb") as f:
        text = analyze_image_with_vision_llm(f)

    print("Vision LLM Extracted Content:\n")
    print(text)


def test_unified_processor():
    print("\n🔄 Testing Unified process_uploaded_file()...\n")

    for filename in os.listdir(TEST_FILES_DIR):
        file_path = os.path.join(TEST_FILES_DIR, filename)

        print(f"\nTesting file: {filename}")

        with open(file_path, "rb") as f:
            class DummyFile:
                def __init__(self, file_obj, name):
                    self.file = file_obj
                    self.type = get_mime_type(name)

                def read(self):
                    return self.file.read()

                def seek(self, pos):
                    self.file.seek(pos)

            dummy = DummyFile(f, filename)

            result = process_uploaded_file(dummy)

        print("Output Preview:")
        print(result[:500])
        print("-" * 50)


def get_mime_type(filename):
    if filename.endswith(".pdf"):
        return "application/pdf"
    elif filename.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        return "image/jpeg"
    else:
        return "application/octet-stream"


if __name__ == "__main__":

    print("===================================")
    print("🧪 Testing multimodal_processor.py")
    print("===================================")

    test_pdf()
    '''test_docx()'''
    test_image_ocr()
    test_image_vision()
    '''test_unified_processor()'''
