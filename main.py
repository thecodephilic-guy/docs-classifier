from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
from dataset_loader import CATEGORIES
from ocr_extraction import extract_invoice_data, extract_text

# Load trained model
model = tf.keras.models.load_model("document_classifier.h5")

# Define FastAPI app
app = FastAPI()

# Preprocess function
def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    # Keep original image for OCR
    original_img = img.copy()
    
    # Preprocess for model
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized.astype(np.float32) / 255.0  # Explicitly convert to float32
    return original_img, np.expand_dims(img_normalized, axis=0)

@app.get("/")
async def home_route():
        return {"Hello from home"}

@app.post("/classify_and_extract/")
async def classify_image(file: UploadFile = File(description="Image file to classify")):
    try:
        image_data = await file.read()

        if not file.filename:
            return {"error": "No filename provided"}

        if len(image_data) == 0:
            return {"error": "Empty file received"}

        if not file.content_type or not file.content_type.startswith('image/'):
            return {"error": "File must be an image"}

        image, img_array = preprocess_image(image_data)
        
        # Make prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        class_name = CATEGORIES[class_index]

        if class_name == "invoice":
             extracted_data = extract_invoice_data(image)
        else:
             extracted_data = extract_text(image)
        
        return {"class": class_name, "extracted_data": extracted_data}
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}