import pytesseract
import cv2
import re
import numpy as np
from dataset_loader import CATEGORIES

# (Windows Only) Set path for Tesseract
# Uncomment & modify if Tesseract is not detected automatically
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to preprocess images for OCR
def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Apply thresholding
    return thresh

# Function to extract plain text
def extract_text(image):
    processed = preprocess_for_ocr(image)
    text = pytesseract.image_to_string(processed)
    return text.strip()

# Function to extract structured invoice data
def extract_invoice_data(image):
    text = extract_text(image)
    
    # Define regex patterns to extract information
    invoice_number = re.search(r"Invoice\s*No[:\-]?\s*(\S+)", text, re.IGNORECASE)
    date = re.search(r"Date[:\-]?\s*([\d\/\-]+)", text, re.IGNORECASE)
    amount = re.search(r"Total[:\-]?\s*\$?([\d,.]+)", text, re.IGNORECASE)
    vendor = re.search(r"Vendor[:\-]?\s*([^\n]+)", text, re.IGNORECASE)

    return {
        "Invoice Number": invoice_number.group(1) if invoice_number else "Not found",
        "Date": date.group(1) if date else "Not found",
        "Total Amount": amount.group(1) if amount else "Not found",
        "Vendor": vendor.group(1) if vendor else "Not found",
        "Full Text": text
    }