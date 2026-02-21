import tensorflow as tf
from keras import layers, models, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
import json

warnings.filterwarnings('ignore')

class FixedDrowsinessDetector:
    def __init__(self, img_size=96, detection_type='eyes'):
        """
        Fixed drowsiness detection with consistent label mapping and proper preprocessing
        """
        self.img_size = img_size
        self.detection_type = detection_type
        self.model = None
        self.history = None
        self.class_names = []
        self.label_mapping = {}  # Store the actual mapping used during training
        
        print(f"Fixed Drowsiness Detector initialized")
        print(f"Image size: {img_size}x{img_size}")
        print(f"Detection type: {detection_type.upper()}")
    
    def verify_and_map_labels(self, data_dir, is_training=True):
        """
        CRITICAL: Verify directory structure and create consistent label mapping
        """
        print(f"\nVERIFYING LABEL MAPPING FOR: {data_dir}")
        print("="*50)
        
        if not os.path.exists(data_dir):
            print(f"Directory not found: {data_dir}")
            return None
        
        # Get class directories in alphabetical order (this is crucial)
        class_dirs = sorted([d for d in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, d))])
        
        if not class_dirs:
            print(f"No class directories found in {data_dir}")
            return None
        
        # Create/verify label mapping
        if is_training:
            # During training, create the mapping based on alphabetical order
            self.class_names = class_dirs
            self.label_mapping = {}
            
            print(f"Found {len(class_dirs)} classes:")
            for i, class_name in enumerate(class_dirs):
                class_path = os.path.join(data_dir, class_name)
                image_count = len([f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                
                # tf.keras assigns labels alphabetically: first dir = 0, second dir = 1
                tf_label = i
                self.label_mapping[class_name] = tf_label
                
                print(f"   {i}: '{class_name}' -> Label {tf_label} ({image_count} images)")
            
            # Save mapping to file for future reference
            self.save_label_mapping()
            
        else:
            # During testing, verify consistency with training mapping
            if not self.label_mapping:
                print("No training label mapping found. Loading from file...")
                if not self.load_label_mapping():
                    print("Cannot proceed without label mapping")
                    return None
            
            print(f"Verifying consistency with training mapping:")
            print(f"   Training classes: {list(self.label_mapping.keys())}")
            print(f"   Current directory classes: {class_dirs}")
        
        # Verify the semantic meaning of labels
        print(f"\nSEMANTIC LABEL VERIFICATION:")
        for class_name, label in self.label_mapping.items():
            if self.detection_type.lower() == 'eyes':
                if any(word in class_name.lower() for word in ['closed', 'close', 'drowsy', 'sleepy']):
                    expected_meaning = "CLOSED EYES (DROWSY)"
                elif any(word in class_name.lower() for word in ['open', 'opened', 'alert', 'awake']):
                    expected_meaning = "OPEN EYES (ALERT)"
                else:
                    expected_meaning = "UNKNOWN - PLEASE VERIFY"
            else:  # yawn detection
                if any(word in class_name.lower() for word in ['yawn', 'yawning', 'drowsy']):
                    expected_meaning = "YAWN (DROWSY)"
                elif any(word in class_name.lower() for word in ['no_yawn', 'not_yawning', 'alert']):
                    expected_meaning = "NO YAWN (ALERT)"
                else:
                    expected_meaning = "UNKNOWN - PLEASE VERIFY"
            
            print(f"   '{class_name}' -> Label {label} ({expected_meaning})")
        
        return self.label_mapping
    
    def save_label_mapping(self):
        """Save label mapping to JSON file"""
        try:
            mapping_file = f"label_mapping_{self.detection_type}.json"
            mapping_data = {
                'detection_type': self.detection_type,
                'img_size': self.img_size,
                'class_names': self.class_names,
                'label_mapping': self.label_mapping
            }
            
            with open(mapping_file, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            
            print(f"Label mapping saved to: {mapping_file}")
            
        except Exception as e:
            print(f"Could not save label mapping: {e}")
    
    def load_label_mapping(self):
        """Load label mapping from JSON file"""
        try:
            mapping_file = f"label_mapping_{self.detection_type}.json"
            
            if not os.path.exists(mapping_file):
                print(f"Label mapping file not found: {mapping_file}")
                return False
            
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            self.class_names = mapping_data['class_names']
            self.label_mapping = mapping_data['label_mapping']
            
            print(f"Label mapping loaded from: {mapping_file}")
            return True
            
        except Exception as e:
            print(f"Could not load label mapping: {e}")
            return False
    
    def create_dataset(self, data_dir, batch_size=32, validation_split=0.2, is_training=True):
        """Create dataset with verified label mapping"""
        
        # First, verify the label mapping
        if not self.verify_and_map_labels(data_dir, is_training):
            return None
        
        try:
            if is_training:
                # For training, use the standard approach but verify labels
                train_ds = tf.keras.utils.image_dataset_from_directory(
                    data_dir, validation_split=validation_split, subset="training",
                    seed=123, image_size=(self.img_size, self.img_size),
                    batch_size=batch_size, label_mode='binary'
                )
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    data_dir, validation_split=validation_split, subset="validation", 
                    seed=123, image_size=(self.img_size, self.img_size),
                    batch_size=batch_size, label_mode='binary'
                )
                
                # Verify TensorFlow's automatic label assignment matches our expectation
                tf_class_names = train_ds.class_names
                print(f"\nTensorFlow class assignment:")
                for i, class_name in enumerate(tf_class_names):
                    print(f"   TF Label {i}: '{class_name}'")
                    if class_name in self.label_mapping:
                        expected_label = self.label_mapping[class_name]
                        if i != expected_label:
                            print(f"MISMATCH! Expected {expected_label}, got {i}")
                        else:
                            print(f"   Matches our mapping")
                
                return train_ds, val_ds
                
            else:
                # For testing, create dataset without shuffling
                dataset = tf.keras.utils.image_dataset_from_directory(
                    data_dir, seed=123, image_size=(self.img_size, self.img_size),
                    batch_size=batch_size, label_mode='binary', shuffle=False
                )
                
                return dataset
                
        except Exception as e:
            print(f"tf.keras.utils.image_dataset_from_directory failed: {e}")
            print("Falling back to manual dataset creation...")
            return self.create_manual_dataset(data_dir, batch_size, validation_split, is_training)
    
    def create_manual_dataset(self, data_dir, batch_size, validation_split, is_training):
        """Manual dataset creation with explicit label control and FIXED preprocessing"""
        import pathlib
        from sklearn.model_selection import train_test_split
        
        print("Creating dataset manually with explicit label mapping...")
        
        # Use our verified label mapping instead of relying on TensorFlow's automatic assignment
        all_image_paths = []
        all_labels = []
        
        for class_name, label in self.label_mapping.items():
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Class directory not found: {class_path}")
                continue
            
            # Get all images in this class
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            class_images = []
            
            for ext in image_extensions:
                class_images.extend([
                    os.path.join(class_path, f) for f in os.listdir(class_path)
                    if f.lower().endswith(ext.lower())
                ])
            
            # Add to dataset with our explicit label
            all_image_paths.extend(class_images)
            all_labels.extend([label] * len(class_images))
            
            print(f"   Class '{class_name}' -> Label {label}: {len(class_images)} images")
        
        print(f"Total images: {len(all_image_paths)}")
        print(f"Label distribution: {np.bincount(all_labels)}")
        
        # Convert to numpy arrays
        all_image_paths = np.array(all_image_paths)
        all_labels = np.array(all_labels)
        
        if is_training:
            # Split into train and validation
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                all_image_paths, all_labels, 
                test_size=validation_split, 
                random_state=123, 
                stratify=all_labels
            )
            
            train_dataset = self._create_tf_dataset(train_paths, train_labels, batch_size, is_training=True)
            val_dataset = self._create_tf_dataset(val_paths, val_labels, batch_size, is_training=False)
            
            return train_dataset, val_dataset
        else:
            dataset = self._create_tf_dataset(all_image_paths, all_labels, batch_size, is_training=False)
            return dataset
    
    def _create_tf_dataset(self, image_paths, labels, batch_size, is_training=True):
        """Create TensorFlow dataset from paths and labels with FIXED preprocessing"""
        
        def load_and_preprocess_image(path, label):
            # Load image
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, [self.img_size, self.img_size])
            
            # CRITICAL FIX: Don't normalize here since model has Rescaling layer
            # The model's first layer will handle normalization (divide by 255)
            image = tf.cast(image, tf.float32)  # Keep values in [0, 255] range
            
            # Data augmentation for training only
            if is_training:
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, 0.1)
                image = tf.image.random_contrast(image, 0.9, 1.1)
            
            return image, tf.cast(label, tf.float32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(image_paths))
        
        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def build_model(self):
        """Build improved model with proper normalization"""
        model = models.Sequential([
            layers.Input(shape=(self.img_size, self.img_size, 3)),
            
            # IMPORTANT: This layer handles normalization - input should be [0, 255]
            layers.Rescaling(1./255),
            
            # Data augmentation during training
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            
            # Convolution blocks
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
            
            # Classification layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        print("Model built successfully")
        return model
    
    def train(self, train_dir, validation_dir=None, epochs=30, batch_size=16):
        """Train with consistent label mapping"""
        print(f"\n{'='*50}")
        print(f"TRAINING {self.detection_type.upper()} DETECTOR WITH FIXED LABELS")
        print(f"{'='*50}")
        
        # Create datasets with verified labels
        if validation_dir and os.path.exists(validation_dir):
            print(f"Using separate validation directory: {validation_dir}")
            train_data, _ = self.create_dataset(train_dir, batch_size, is_training=True)
            val_data = self.create_dataset(validation_dir, batch_size, is_training=False)
        else:
            print("Using train/validation split from training directory")
            train_data, val_data = self.create_dataset(train_dir, batch_size, validation_split=0.25, is_training=True)
        
        if train_data is None:
            print("Failed to create training dataset")
            return None
        
        # Build model
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-7, verbose=1)
        ]
        
        # Train
        print(f"Training with verified label mapping:")
        for class_name, label in self.label_mapping.items():
            meaning = self.get_label_meaning(label)
            print(f"   '{class_name}' -> {label} ({meaning})")
        
        self.history = self.model.fit(
            train_data, 
            epochs=epochs, 
            validation_data=val_data,
            callbacks=callbacks, 
            verbose=1
        )
        
        # Plot training results immediately after training
        self.plot_training_results()
        
        return self.history
    
    def plot_training_results(self):
        """Plot training results after training completes"""
        if not self.history:
            print("No training history available")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        train_acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        epochs = range(1, len(train_acc) + 1)
        
        ax1.plot(epochs, train_acc, 'b-', label='Training', linewidth=2)
        ax1.plot(epochs, val_acc, 'r-', label='Validation', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(self.history.history['loss'], 'b-', label='Training', linewidth=2)
        ax2.plot(self.history.history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in self.history.history:
            ax3.plot(self.history.history['precision'], 'b-', label='Training', linewidth=2)
            ax3.plot(self.history.history['val_precision'], 'r-', label='Validation', linewidth=2)
            ax3.set_title('Model Precision', fontsize=14)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Precision')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in self.history.history:
            ax4.plot(self.history.history['recall'], 'b-', label='Training', linewidth=2)
            ax4.plot(self.history.history['val_recall'], 'r-', label='Validation', linewidth=2)
            ax4.set_title('Model Recall', fontsize=14)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Recall')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training Results - {self.detection_type.upper()} Detection', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Training summary
        best_val_acc = max(val_acc)
        best_epoch = val_acc.index(best_val_acc) + 1
        final_train_acc = train_acc[-1]
        final_val_acc = val_acc[-1]
        overfitting_gap = final_train_acc - final_val_acc
        
        print(f"\nTraining Summary:")
        print(f"   Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
        print(f"   Final Training Accuracy: {final_train_acc:.4f}")
        print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"   Overfitting Gap: {overfitting_gap:.4f}")
        
        if overfitting_gap < 0.05:
            print("   Model is well-generalized")
        elif overfitting_gap < 0.1:
            print("   Slight overfitting, but acceptable")
        else:
            print("   Significant overfitting detected")
    
    def get_label_meaning(self, label):
        """Get human-readable meaning of label"""
        if self.detection_type.lower() == 'eyes':
            return "CLOSED EYES (DROWSY)" if label == 0 else "OPEN EYES (ALERT)"
        else:
            return "NO YAWN (ALERT)" if label == 0 else "YAWN (DROWSY)"
    
    def evaluate(self, test_dir, batch_size=32):
        """Evaluate with consistent label mapping and show confusion matrix"""
        if not self.model:
            print("No trained model available")
            return
        
        print(f"\nEVALUATING MODEL WITH VERIFIED LABELS")
        print(f"{'='*40}")
        
        # Create test dataset with verified labels
        test_data = self.create_dataset(test_dir, batch_size, is_training=False)
        if test_data is None:
            print("Failed to create test dataset")
            return
        
        # Evaluate
        results = self.model.evaluate(test_data, verbose=1)
        
        print(f"\nTest Results:")
        for name, value in zip(self.model.metrics_names, results):
            print(f"   {name.title()}: {value:.4f}")
        
        # Generate predictions for confusion matrix
        try:
            print("Generating predictions for confusion matrix...")
            predictions = self.model.predict(test_data)
            y_pred = (predictions > 0.5).astype(int).flatten()
            
            # Get true labels - FIXED VERSION
            y_true = []
            for batch_images, batch_labels in test_data:
                batch_labels_np = batch_labels.numpy()
                # Handle different shapes that might occur
                if batch_labels_np.ndim > 1:
                    batch_labels_np = batch_labels_np.flatten()
                y_true.extend(batch_labels_np.astype(int))
            
            # Convert to flat numpy array and ensure it's 1D
            y_true = np.array(y_true).flatten()
            
            print(f"Debug info:")
            print(f"   y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
            print(f"   y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
            print(f"   y_true sample: {y_true[:10] if len(y_true) >= 10 else y_true}")
            print(f"   y_pred sample: {y_pred[:10] if len(y_pred) >= 10 else y_pred}")
            
            # Ensure same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            print(f"\nPrediction Analysis:")
            print(f"   Total samples: {len(y_true)}")
            print(f"   True label distribution: {np.bincount(y_true)}")
            print(f"   Predicted label distribution: {np.bincount(y_pred)}")
            
            # Show what each label means
            print(f"\nLabel meanings:")
            print(f"   Label 0: {self.get_label_meaning(0)}")
            print(f"   Label 1: {self.get_label_meaning(1)}")
            
            # Confusion matrix
            self.plot_confusion_matrix(y_true, y_pred)
            
            # Classification report
            print(f"\nClassification Report:")
            plot_labels = [self.get_label_meaning(i) for i in range(2)]
            print(classification_report(y_true, y_pred, target_names=plot_labels))
            
        except Exception as e:
            print(f"Error generating detailed analysis: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        
        # Create meaningful labels for the plot
        plot_labels = [self.get_label_meaning(i) for i in range(2)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=plot_labels, yticklabels=plot_labels)
        plt.title(f'Confusion Matrix - {self.detection_type.upper()} Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.15, 0.02, f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                   fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def save_model(self, filepath=None):
        """Save model with label mapping"""
        if not self.model:
            print("No model to save")
            return None
        
        if filepath is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"{self.detection_type}_fixed_model_{timestamp}.h5"
        
        try:
            # Save model
            self.model.save(filepath)
            print(f"Model saved: {filepath}")
            
            # Save label mapping alongside model
            self.save_label_mapping()
            
            return filepath
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return None
    
    def load_model(self, filepath):
        """Load model and its label mapping"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded: {filepath}")
            
            # Try to load corresponding label mapping
            if self.load_label_mapping():
                print("Label mapping loaded successfully")
            else:
                print("Could not load label mapping - may cause issues")
            
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


def main_fixed():
    """Main function with label consistency fixes and proper preprocessing"""
    
    print("FIXED DROWSINESS DETECTION - PROPER PREPROCESSING")
    print("="*60)
    
    # Configuration
    DETECTION_TYPE = 'eyes'
    IMG_SIZE = 96
    BATCH_SIZE = 16
    EPOCHS = 20
    
    # Dataset paths - UPDATE THESE
    TRAIN_DIR = r'C:\Users\haniszaim\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\DATASET\DROWSINESS\EYES_CLOSED_OPENEDV4\train'
    VAL_DIR = r'C:\Users\haniszaim\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\DATASET\DROWSINESS\EYES_CLOSED_OPENEDV4\val'
    TEST_DIR = r'C:\Users\haniszaim\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\DATASET\DROWSINESS\EYES_CLOSED_OPENEDV4\test'
    
    if not os.path.exists(TRAIN_DIR):
        print(f"Training directory not found: {TRAIN_DIR}")
        return
    
    try:
        # Initialize fixed detector
        detector = FixedDrowsinessDetector(IMG_SIZE, DETECTION_TYPE)
        
        # Train with label verification and fixed preprocessing
        print("\nStarting training with verified labels and proper preprocessing...")
        
        # Check if validation directory exists
        validation_dir = VAL_DIR if os.path.exists(VAL_DIR) else None
        if validation_dir:
            print(f"Using separate validation directory: {validation_dir}")
        else:
            print("Validation directory not found, using train/validation split")
        
        history = detector.train(TRAIN_DIR, validation_dir=validation_dir, epochs=EPOCHS, batch_size=BATCH_SIZE)
        
        if history is None:
            print("Training failed")
            return
        
        # Save model with label mapping
        saved_model_path = detector.save_model()
        
        # Test if test directory exists
        if os.path.exists(TEST_DIR):
            print("\nTesting model with consistent labels...")
            detector.evaluate(TEST_DIR, BATCH_SIZE)
        
        print(f"\nTraining and testing completed!")
        print(f"Model saved: {saved_model_path}")
        print(f"Label mapping saved: label_mapping_{DETECTION_TYPE}.json")
        
        print(f"\nKey fixes implemented:")
        print(f"   Fixed double normalization issue")
        print(f"   Added training plots and confusion matrix")
        print(f"   Consistent label mapping throughout")
        print(f"   Proper preprocessing pipeline")
        print(f"   Support for separate validation directory")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_fixed()