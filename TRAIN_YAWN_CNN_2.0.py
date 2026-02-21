"""
YAWN DETECTION TRAINING - FIXED AND OPTIMIZED
Ensures correct label mapping and preprocessing
"""

import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import json

tf.get_logger().setLevel('ERROR')

# ============== CONFIGURATION ==============
IMG_SIZE = 96
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0005

# YOUR DATASET PATHS - UPDATE THESE
TRAIN_DIR = r'C:\Users\USER\Desktop\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\DATASET\DROWSINESS\YAWNING_SPLIT\train'
VAL_DIR = r'C:\Users\USER\Desktop\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\DATASET\DROWSINESS\YAWNING_SPLIT\val'
TEST_DIR = r'C:\Users\USER\Desktop\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\DATASET\DROWSINESS\YAWNING_SPLIT\test'

# ============== GPU SETUP ==============
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU available: {len(gpus)} GPU(s)")
        print(f"  {tf.config.list_physical_devices('GPU')}")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("⚠ No GPU detected - using CPU")

# ============== VERIFY LABEL MAPPING ==============
def verify_dataset_structure(data_dir):
    """Check folder structure and determine label mapping"""
    print(f"\n{'='*60}")
    print(f"VERIFYING DATASET: {os.path.basename(data_dir)}")
    print(f"{'='*60}")
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return None
    
    # Get class folders (alphabetically sorted - THIS IS HOW TF ASSIGNS LABELS!)
    class_folders = sorted([d for d in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, d))])
    
    if len(class_folders) != 2:
        print(f"❌ Expected 2 folders, found {len(class_folders)}")
        return None
    
    print(f"\nFound {len(class_folders)} classes:")
    label_mapping = {}
    
    for idx, folder in enumerate(class_folders):
        folder_path = os.path.join(data_dir, folder)
        image_count = len([f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        # TensorFlow assigns labels alphabetically
        label_mapping[folder] = idx
        
        # Determine what this label means
        if 'no' in folder.lower() or 'not' in folder.lower():
            meaning = "NO YAWN (output should be LOW ~0.0-0.2)"
        else:
            meaning = "YAWN (output should be HIGH ~0.8-1.0)"
        
        print(f"  [{idx}] '{folder}' → Label {idx}")
        print(f"      {image_count} images | {meaning}")
    
    print(f"\n⚠ CRITICAL: TensorFlow assigns labels ALPHABETICALLY!")
    print(f"  Your folders: {class_folders}")
    print(f"  Label 0 = {class_folders[0]}")
    print(f"  Label 1 = {class_folders[1]}")
    
    return label_mapping

# ============== CREATE MODEL ==============
def create_yawn_model(img_size=96):
    """Create CNN model matching your existing architecture"""
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(img_size, img_size, 3)),
        
        # IMPORTANT: Rescaling layer - expects input [0, 255]
        layers.Rescaling(1./255),
        
        # Data augmentation (only active during training)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        
        # Conv Block 1
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        
        # Conv Block 2
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        
        # Conv Block 3
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        
        # Conv Block 4
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer - sigmoid for binary classification
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

# ============== CREATE DATASETS ==============
def create_datasets(train_dir, val_dir, batch_size, img_size):
    """Create training and validation datasets"""
    
    print(f"\n{'='*60}")
    print("CREATING DATASETS")
    print(f"{'='*60}")
    
    # Training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='binary'
    )
    
    # Validation dataset
    if val_dir and os.path.exists(val_dir):
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            seed=123,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode='binary'
        )
        print("✓ Using separate validation directory")
    else:
        print("⚠ No validation directory - splitting from training data")
        val_split = 0.2
        val_batches = int(len(train_ds) * val_split)
        val_ds = train_ds.take(val_batches)
        train_ds = train_ds.skip(val_batches)
    
    # Verify class names
    print(f"\nTensorFlow's class assignment:")
    for idx, name in enumerate(train_ds.class_names):
        print(f"  Label {idx}: '{name}'")
    
    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

# ============== TRAINING ==============
def train_model(model, train_ds, val_ds, epochs):
    """Train the model"""
    
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    
    # Create timestamp for saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'yawn_model_best_{timestamp}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, timestamp

# ============== PLOT RESULTS ==============
def plot_training_results(history):
    """Plot training metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, weight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, weight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
    axes[1, 0].set_title('Precision', fontsize=14, weight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
    axes[1, 1].set_title('Recall', fontsize=14, weight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Yawn Detection Training Results', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('yawn_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = max(history.history['val_accuracy'])
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Final Training Accuracy:  {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Overfitting Gap: {final_train_acc - final_val_acc:.4f}")

# ============== EVALUATE ==============
def evaluate_model(model, test_dir, batch_size, img_size):
    """Evaluate on test set"""
    
    if not os.path.exists(test_dir):
        print(f"⚠ Test directory not found: {test_dir}")
        return
    
    print(f"\n{'='*60}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*60}")
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )
    
    results = model.evaluate(test_ds, verbose=1)
    
    print(f"\nTest Results:")
    for name, value in zip(model.metrics_names, results):
        print(f"  {name.title()}: {value:.4f}")
    
    # Get predictions
    predictions = model.predict(test_ds)
    y_pred = (predictions > 0.5).astype(int).flatten()
    
    # Get true labels - fixed version
    y_true = []
    for images, labels in test_ds:
        y_true.extend(labels.numpy().flatten())
    y_true = np.array(y_true).astype(int)
    
    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    print(f"\nPrediction Distribution:")
    print(f"  True labels:  {np.bincount(y_true)}")
    print(f"  Predictions:  {np.bincount(y_pred)}")
    print(f"  Accuracy: {np.mean(y_true == y_pred):.4f}")
    
    # Show confusion info
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  [[no_yawn→no_yawn  no_yawn→yawn]")
    print(f"   [yawn→no_yawn     yawn→yawn]]")
    print(f"  {cm}")

# ============== SAVE MODEL ==============
def save_final_model(model, timestamp):
    """Save model with metadata"""
    
    model_path = f'yawn_model_final_{timestamp}.h5'
    model.save(model_path)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'img_size': IMG_SIZE,
        'architecture': 'CNN Binary Classification',
        'label_0': 'no_yawn (LOW output ~0.0-0.2)',
        'label_1': 'yawn (HIGH output ~0.8-1.0)',
        'preprocessing': 'Rescaling(1./255) - expects [0, 255] input',
        'threshold': 0.5
    }
    
    with open(f'yawn_model_metadata_{timestamp}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("MODEL SAVED")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Metadata: yawn_model_metadata_{timestamp}.json")
    
    return model_path

# ============== MAIN ==============
def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("YAWN DETECTION MODEL TRAINING")
    print("="*60)
    
    # Verify datasets
    train_mapping = verify_dataset_structure(TRAIN_DIR)
    if train_mapping is None:
        return
    
    if VAL_DIR and os.path.exists(VAL_DIR):
        verify_dataset_structure(VAL_DIR)
    
    # Create datasets
    train_ds, val_ds = create_datasets(TRAIN_DIR, VAL_DIR, BATCH_SIZE, IMG_SIZE)
    
    # Create model
    print(f"\nCreating model...")
    model = create_yawn_model(IMG_SIZE)
    model.summary()
    
    # Train
    history, timestamp = train_model(model, train_ds, val_ds, EPOCHS)
    
    # Plot results
    plot_training_results(history)
    
    # Evaluate on test set
    if TEST_DIR and os.path.exists(TEST_DIR):
        evaluate_model(model, TEST_DIR, BATCH_SIZE, IMG_SIZE)
    
    # Save final model
    model_path = save_final_model(model, timestamp)
    
    print(f"\n{'='*60}")
    print("✓ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Use this model in your detection system:")
    print(f"  {model_path}")
    print(f"\nExpected behavior:")
    print(f"  no_yawn → LOW output (0.0-0.2)")
    print(f"  yawn → HIGH output (0.8-1.0)")

if __name__ == "__main__":
    main()