import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define dataset path
DATASET_PATH = "dataset"

# Image dimensions
IMG_SIZE = 128  # Resize all images to 128x128
CATEGORIES = ["handwritten", "invoice", "email"] 

# Data storage
X, y = [], []

# Load images from folders
for category in CATEGORIES:
    folder_path = os.path.join(DATASET_PATH, category)
    label = CATEGORIES.index(category)  # Convert category to numeric label
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)  # Load image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 128x128
        X.append(img)
        y.append(label)

# Convert lists to numpy arrays
X = np.array(X) / 255.0  # Normalize pixel values (0-1)
y = np.array(y)

# One-hot encode labels
y = to_categorical(y, num_classes=len(CATEGORIES))

# Split dataset into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset Loaded: {len(X_train)} training, {len(X_test)} testing images")