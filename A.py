import streamlit as st
import base64
import json
import io
import os
import fitz  # PyMuPDF
import openai
import easyocr
import numpy as np
from PIL import Image
 
# ----------------------------------------
# Setup API keys (Replace with your actual keys)
# ----------------------------------------
openai.api_key =  "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" # Replace with your OpenAI API key
 
# Create an EasyOCR reader object for English (you can add more languages if needed)
reader = easyocr.Reader(['en'], gpu=False)
 
# Ensure the folder for storing images exists
IMAGE_FOLDER = "pdf_images"
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
 
# ----------------------------------------
# AGENT 1: PDF Processing & Image Conversion using PyMuPDF
# ----------------------------------------
def agent1_process_pdf(uploaded_pdf):
    """
    Reads an uploaded PDF using PyMuPDF and converts each page into an image.
    Each image is saved into the folder 'pdf_images' and also encoded in base64 along with page metadata.
    The DPI is set to 350 for improved image clarity.
    """
    pdf_bytes = uploaded_pdf.read()
    try:
        # Open the PDF from bytes using PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        st.error(f"Error processing PDF {uploaded_pdf.name}: {e}")
        return None
 
    # Extract the base file name without extension
    base_filename = os.path.splitext(uploaded_pdf.name)[0]
   
    pages_data = []
    for i, page in enumerate(doc, start=1):
        try:
            # Render page to an image with lower DPI (72) to reduce size
            pix = page.get_pixmap(dpi=350)
            img_bytes = pix.tobytes("png")
           
            # Save the image file to the pdf_images folder
            file_path = os.path.join(IMAGE_FOLDER, f"{base_filename}_page_{i}.png")
            with open(file_path, "wb") as f:
                f.write(img_bytes)
           
            # Encode the image in base64 for further processing
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            pages_data.append({"page_number": i, "base64_image": img_b64})
        except Exception as e:
            st.error(f"Error converting page {i} in {uploaded_pdf.name}: {e}")
            continue
 
    document_data = {"document_name": uploaded_pdf.name, "pages": pages_data}
    return document_data
 
# ----------------------------------------
# AGENT 2: Text Extraction using EasyOCR
# ----------------------------------------
def agent2_extract_text(document_data):
    """
    For each base64-encoded page image in document_data, this function:
      1. Decodes the image.
      2. Converts it to a numpy array.
      3. Uses EasyOCR to extract text.
    The extracted text from all pages is concatenated for further processing.
    """
    extracted_pages = []
    for page in document_data["pages"]:
        try:
            # Decode the base64 image and open it with PIL
            img_bytes = base64.b64decode(page["base64_image"])
            pil_img = Image.open(io.BytesIO(img_bytes))
            # Convert the PIL image to a numpy array
            img_np = np.array(pil_img)
            # Use EasyOCR to extract text; detail=0 returns a list of text strings
            text_list = reader.readtext(img_np, detail=0)
            page_text = " ".join(text_list)
        except Exception as e:
            st.error(f"Error extracting text on page {page['page_number']}: {e}")
            continue
 
        extracted_pages.append({
            "page_number": page["page_number"],
            "extracted_text": page_text
        })
 
    combined_document_text = "\n".join([p["extracted_text"] for p in extracted_pages])
    document_text_data = {
        "document_name": document_data["document_name"],
        "pages": extracted_pages,
        "combined_text": combined_document_text
    }
    return document_text_data
 
# ----------------------------------------
# AGENT 3: Information Extraction using OpenAI API in Plain Text
# ----------------------------------------
def agent3_extract_details(document_text_data):
    """
    Uses the OpenAI API to first correct any OCR errors in the combined text, then determine
    the document type (Offer Letter, Resume, Invoice, Report, Article, or Letter), and finally
    extract key details based on the identified type. The output is produced in plain text format
    following a strict template.
    """
    combined_text = document_text_data["combined_text"]
    prompt = f"""
You are an AI data extraction assistant. The following text is extracted from a document using OCR and may contain errors. First, correct any obvious OCR mistakes (e.g., fix misspellings or misrecognized characters). Next, determine the document category from these options: Offer Letter, Resume, Invoice, Report, Article, or Letter. Then, based on the identified type, extract and output the key details in plain text format following the template below:
 
If the document is an Offer Letter:
Document Type: Offer Letter
Company Name: <company name>
Company Contacts: <contact1>, <contact2>, ...
Company Address: <company address>
Employee Name: <employee name>
Employee Designation: <designation>
Employee Joining Date: <YYYY-MM-DD>
Employee Salary: <salary details>
Employee Bond Details: <bond details or "None">
 
If the document is a Resume:
Document Type: Resume
Candidate Name: <candidate name>
Contact Information: <contact details>
Skills: <list of skills>
Experience: <summary of experience>
Education: <education details>
Certifications: <certifications or "None">
 
If the document is an Invoice:
Document Type: Invoice
Invoice Number: <invoice number>
Invoice Date: <YYYY-MM-DD>
Vendor Name: <vendor name>
Vendor Contact: <vendor contact details>
Items/Services: <list of items or services>
Total Amount: <total amount>
Due Date: <due date>
Tracking Number: <tracking number>
Paid By: <mode of payment>
Order ID: <order id>
 
 
If the document is a Report:
Document Type: Report
Title: <report title>
Author: <author name>
Date: <YYYY-MM-DD>
Summary: <summary>
Key Findings: <key findings>
Conclusion: <conclusion>
 
If the document is an Article:
Document Type: Article
Title: <article title>
Author: <author name>
Publication Date: <YYYY-MM-DD>
Summary: <summary>
Keywords: <list of keywords>
 
If the document is a Letter:
Document Type: Letter
Sender Name: <sender name>
Recipient Name: <recipient name>
Date: <YYYY-MM-DD>
Subject: <subject>
Message Body: <body content>
 
Output only in plain text, strictly following the format for the identified document type.
Here is the OCR extracted text:
{combined_text}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Replace with "gpt40mini" or "gpt-4" if needed
            messages=[
                {"role": "system", "content": "You are a data extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=8192
        )
        result_text = response.choices[0].message.content.strip()
        return result_text
    except Exception as e:
        return f"Error extracting details: {e}"
 
# ----------------------------------------
# FINAL AGGREGATOR: Process All PDFs
# ----------------------------------------
def process_all_pdfs(uploaded_files):
    """
    Processes all uploaded PDFs by:
      1. Converting each PDF to images and storing them.
      2. Extracting text from each image using EasyOCR.
      3. Extracting key details using the OpenAI API.
    Aggregates the outputs into a single final result.
    """
    processed_documents = []
    for uploaded_pdf in uploaded_files:
        st.info(f"Processing: {uploaded_pdf.name}")
        # Agent 1: Convert PDF to images (and store them)
        document_data = agent1_process_pdf(uploaded_pdf)
        if document_data is None:
            continue
 
        # Agent 2: Extract text using EasyOCR
        document_text_data = agent2_extract_text(document_data)
 
        # Agent 3: Extract structured details using OpenAI API
        extracted_details = agent3_extract_details(document_text_data)
        processed_documents.append(extracted_details)
 
    final_output = {"processed_documents": processed_documents}
    return final_output
 
# ----------------------------------------
# STREAMLIT UI
# ----------------------------------------
st.title("Multi-Agent Text Extractor")
st.write(
    """
    """
)
 
uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
 
if uploaded_files:
    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            final_result = process_all_pdfs(uploaded_files)
        st.success("Processing complete!")
 
        st.subheader("Extracted Details (Final Output)")
        for doc in final_result["processed_documents"]:
            st.text(doc)
 
        # Provide a download button for the final plain text output
        final_text_str = "\n\n".join(final_result["processed_documents"])
        st.download_button(
            label="Download Extracted Details",
            data=final_text_str,
            file_name="extracted_details.txt",
            mime="text/plain"
        )