"""
Quick script to verify your yawn model is working correctly
"""
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
import os

YAWN_MODEL_PATH = r"C:\Users\USER\Desktop\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\CODING\yawn_model_final_20260126_114325.h5"
TEST_DIR = r'C:\Users\USER\Desktop\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\DATASET\DROWSINESS\YAWNING_SPLIT\test'

def create_cnn_model(img_size=96):
    return models.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

print("Loading model...")
model = create_cnn_model()
model.load_weights(YAWN_MODEL_PATH)

print("\n" + "="*60)
print("TESTING YAWN MODEL ON SAMPLE IMAGES")
print("="*60)

# Get folder names (alphabetically - how TF assigns labels)
folders = sorted([f for f in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, f))])
print(f"\nFolders found: {folders}")
print(f"Label 0 = {folders[0]}")
print(f"Label 1 = {folders[1]}")

for folder in folders:
    folder_path = os.path.join(TEST_DIR, folder)
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:5]
    
    print(f"\n{'-'*60}")
    print(f"Testing: {folder}/")
    print(f"{'-'*60}")
    
    scores = []
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Preprocess EXACTLY like training
        img_resized = cv2.resize(img, (96, 96)).astype(np.float32)
        img_input = np.expand_dims(img_resized, axis=0)
        
        # Predict
        score = model.predict(img_input, verbose=0)[0][0]
        scores.append(score)
        
        expected = "YAWN" if 'yawn' in folder.lower() else "NO_YAWN"
        prediction = "YAWN" if score > 0.5 else "NO_YAWN"
        match = "✓" if expected == prediction else "✗"
        
        print(f"  {match} {img_name:30s} → {score:.4f} (pred: {prediction}, true: {expected})")
    
    if scores:
        avg = np.mean(scores)
        print(f"\n  Average score: {avg:.4f}")
        
        if 'yawn' in folder.lower():
            print(f"  Expected: HIGH (~0.8-1.0)")
            if avg < 0.5:
                print(f"  ⚠ WARNING: Scores are LOW! Model may be inverted or preprocessing is wrong!")
        else:
            print(f"  Expected: LOW (~0.0-0.2)")
            if avg > 0.5:
                print(f"  ⚠ WARNING: Scores are HIGH! Model may be inverted!")

print("\n" + "="*60)
print("If you see warnings above, check:")
print("1. Folder naming (alphabetical order determines labels)")
print("2. Preprocessing consistency between training and inference")
print("="*60)