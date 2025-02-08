# Document Classification Service

This project is a Python-based document classification service that categorizes documents into three classes: **Invoice, Handwritten, and Email**. If the document is classified as an **Invoice**, it extracts relevant fields such as Invoice Number, Date, Amount, and Vendor details.

## Setup Instructions

### 1. Fork and Clone the Repository

Ensure you have **Python 3.11** installed on your system before proceeding.

```sh
# Clone the repository
git clone https://github.com/thecodephilic-guy/docs-classifier.git
cd docs-classifier
```

### 2. Create and Activate a Virtual Environment

```sh
# Create a virtual environment
python3.11 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Required Libraries

```sh
pip install -r requirements.txt
```

### 4. Download and Prepare the Dataset

1. Download the dataset from Kaggle: [Real World Documents Collections](https://www.kaggle.com/datasets/shaz13/real-world-documents-collections)
2. Extract the dataset and rename the directory to **`dataset`** inside the cloned repository.
3. Keep only the following three classes and delete the rest:
   - **invoice**
   - **handwritten**
   - **email**

### 5. Run the Scripts in Order

#### **Load the Dataset**

```sh
python3.11 dataset_loader.py
```

#### **Train the Model**

```sh
python3.11 train_model.py
```

#### **Run the HTTP Server**

```sh
uvicorn main:app --reload
```

## Testing the Model

You can test the model using `curl` commands:

### **Test with an Invoice Image**

```sh
curl -X POST \
  http://127.0.0.1:8000/classify_and_extract/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.jpg"
```

**Expected Response:**

```json
{
  "class": "invoice",
  "extracted_data": {
    "Invoice Number": "Not found",
    "Date": "Not found",
    "Total Amount": "5,500.00",
    "Vendor": "Not found",
    "Full Text": "Acme 87 Queen Street\n4161234567 Hamlet, Ontario\n\nCompany name NaRaRe\nHotels invoice\n\nBilled To Date of Issue Invoice Number Amount Due (CAD)\n\nVandeley Group 06/03/2019 0000005 $5,500.00\n..."
  }
}
```

### **Test with an Email Image**

```sh
curl -X POST \
  http://127.0.0.1:8000/classify_and_extract/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@email.jpg"
```

**Expected Response:**

```json
{
  "class": "email",
  "extracted_data": "Professional Email\n\nYour Street Address\nYour City,\nState Zip\n...\nSincerely yours,\nA. Student"
}
```

## Notes

- Ensure the `dataset` folder is correctly structured before running the scripts.
- The service uses **FastAPI** for API handling.
- Make sure to install dependencies using `requirements.txt`.
- I am not performing data cleaning and preprocessing beacase we don’t have a load of data. Dataset contains jpg files only no tabular data.

##
