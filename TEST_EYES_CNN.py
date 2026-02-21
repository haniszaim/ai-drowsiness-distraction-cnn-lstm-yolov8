import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
from tqdm import tqdm
import warnings
import json
warnings.filterwarnings('ignore')

class DrowsinessModelTester:
    def __init__(self, model_path=None, img_size=96, detection_type='eyes'):
        """
        Comprehensive tester for drowsiness detection models
        Ensures proper label matching and detailed evaluation
        """
        self.model_path = model_path
        self.img_size = img_size
        self.detection_type = detection_type
        self.model = None
        self.class_names = []
        
        # Default label mapping for eyes detection (fallback)
        if detection_type.lower() == 'eyes':
            self.label_mapping = {
                'closed': 0,
                'open': 1,
                'Closed': 0,
                'Open': 1,
                'closed_eyes': 0,
                'open_eyes': 1,
                'drowsy': 0,
                'alert': 1,
                'sleepy': 0,
                'awake': 1
            }
        else:  # yawn detection
            self.label_mapping = {
                'no_yawn': 0,
                'yawn': 1,
                'not_yawning': 0,
                'yawning': 1,
                'alert': 0,
                'drowsy': 1
            }
        
        print(f"Drowsiness Model Tester Initialized")
        print(f"Image Size: {img_size}x{img_size}")
        print(f"Detection Type: {detection_type.upper()}")
    
    def load_and_apply_training_labels(self, mapping_file):
        """Load and apply the exact label mapping used during training"""
        if not os.path.exists(mapping_file):
            print(f"No training label mapping found at: {mapping_file}")
            print("Using fallback label logic...")
            return False
        
        try:
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            # Override with training labels
            self.label_mapping = mapping_data['label_mapping']
            self.class_names = mapping_data['class_names']
            
            print(f"Training label mapping loaded successfully:")
            for class_name, label in self.label_mapping.items():
                meaning = self.get_label_meaning(label)
                print(f"   '{class_name}' -> {label} ({meaning})")
            
            return True
            
        except Exception as e:
            print(f"Error loading training label mapping: {e}")
            return False
    
    def get_label_meaning(self, label):
        """Get human-readable meaning of label"""
        if self.detection_type.lower() == 'eyes':
            return "Closed Eyes (Drowsy)" if label == 0 else "Open Eyes (Alert)"
        else:
            return "No Yawn (Alert)" if label == 0 else "Yawn (Drowsy)"
        
    def load_model(self, model_path=None):
        """Load the trained model"""
        if model_path:
            self.model_path = model_path
            
        if not self.model_path:
            print("No model path provided")
            return False
            
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from: {self.model_path}")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def verify_dataset_structure(self, test_dir):
        """Verify and display dataset structure using training labels"""
        print(f"\nDATASET STRUCTURE VERIFICATION")
        print("="*50)
    
        if not os.path.exists(test_dir):
            print(f"Test directory not found: {test_dir}")
            return False
        
    # Get class directories
        class_dirs = [d for d in os.listdir(test_dir) 
                 if os.path.isdir(os.path.join(test_dir, d))]
        class_dirs.sort()
    
        print(f"Test Directory: {test_dir}")
        print(f"Found {len(class_dirs)} class directories:")
    
        total_images = 0
        class_info = {}
    
        for class_name in class_dirs:
            class_path = os.path.join(test_dir, class_name)
        
        # Count images in class directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend([f for f in os.listdir(class_path) 
                              if f.lower().endswith(ext.lower())])
        
            num_images = len(image_files)
            total_images += num_images
        
        # CRITICAL FIX: Use loaded label mapping directly
            if class_name in self.label_mapping:
                mapped_label = self.label_mapping[class_name]
                print(f"   {class_name} -> Label: {mapped_label} ({num_images} images) [FROM TRAINING MAPPING]")
            else:
                print(f"ERROR: '{class_name}' not found in training label mapping!")
                print(f"Available mappings: {list(self.label_mapping.keys())}")
                return False
        
            class_info[class_name] = {
                'count': num_images,
                'mapped_label': mapped_label
            }
    
        print(f"Total images: {total_images}")
    
    # Verify label mapping makes sense
        print(f"\nLABEL MAPPING VERIFICATION:")
        for class_name, info in class_info.items():
            mapped_label = info['mapped_label']
            meaning = self.get_label_meaning(mapped_label)
            print(f"   {class_name} -> {mapped_label} ({meaning})")
    
        return class_info
    
    def get_mapped_label_fallback(self, class_name):
        """Fallback label mapping if training mapping not available"""
        clean_name = class_name.lower().strip()
        
        # Default mapping based on detection type and common patterns
        if self.detection_type.lower() == 'eyes':
            if any(word in clean_name for word in ['closed', 'close', 'drowsy', 'sleepy', 'tired']):
                return 0  # Closed/Drowsy
            elif any(word in clean_name for word in ['open', 'alert', 'awake', 'active']):
                return 1  # Open/Alert
        else:  # yawn
            if any(word in clean_name for word in ['yawn', 'drowsy', 'sleepy', 'tired']):
                return 1  # Yawning/Drowsy
            elif any(word in clean_name for word in ['no_yawn', 'not_yawning', 'alert', 'awake']):
                return 0  # Not yawning/Alert
        
        print(f"Cannot determine label for class '{class_name}'. Defaulting to 0.")
        return 0
    
    def load_test_data_with_verification(self, test_dir, batch_size=32):
        """Load test data with proper label verification"""
        print(f"\nLOADING TEST DATA WITH LABEL VERIFICATION")
        print("="*50)
        
        # Verify dataset structure first
        class_info = self.verify_dataset_structure(test_dir)
        if not class_info:
            return None, None
        
        # Collect all image paths and their true labels
        image_paths = []
        true_labels = []
        
        for class_name, info in class_info.items():
            class_path = os.path.join(test_dir, class_name)
            mapped_label = info['mapped_label']
            
            # Get all image files in this class
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for filename in os.listdir(class_path):
                if any(filename.lower().endswith(ext.lower()) for ext in image_extensions):
                    image_paths.append(os.path.join(class_path, filename))
                    true_labels.append(mapped_label)
        
        print(f"Total test samples: {len(image_paths)}")
        print(f"Label distribution:")
        unique_labels, counts = np.unique(true_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            meaning = self.get_label_meaning(label)
            print(f"   Label {label} ({meaning}): {count} samples")
        
        return image_paths, true_labels
    
    def preprocess_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            img = cv2.resize(img, (self.img_size, self.img_size))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # DON'T normalize - let the model's Rescaling layer handle it
            img_batch = np.expand_dims(img_rgb.astype(np.float32), axis=0)
        
            return img_batch
        except Exception as e:
            return None
               
    def comprehensive_test(self, test_dir, batch_size=32, save_results=True):
        """Comprehensive testing with detailed analysis"""
        if not self.model:
            print("No model loaded. Use load_model() first.")
            return
            
        print(f"\nCOMPREHENSIVE MODEL TESTING")
        print("="*50)
        
        # Load test data with verification
        image_paths, true_labels = self.load_test_data_with_verification(test_dir, batch_size)
        if image_paths is None:
            return
        
        # Make predictions
        print(f"\nMaking predictions...")
        predictions = []
        failed_images = []
        
        for i, image_path in enumerate(tqdm(image_paths, desc="Predicting")):
            img_batch = self.preprocess_image(image_path)
            if img_batch is not None:
                try:
                    pred = self.model.predict(img_batch, verbose=0)[0][0]
                    predictions.append(pred)
                except Exception as e:
                    print(f"Prediction failed for {image_path}: {e}")
                    predictions.append(0.5)  # Default prediction
                    failed_images.append(image_path)
            else:
                predictions.append(0.5)  # Default for failed images
                failed_images.append(image_path)
        
        if failed_images:
            print(f"{len(failed_images)} images failed to process")
        
        # Convert predictions to binary based on detection type
        predictions = np.array(predictions)
        if self.detection_type.lower() == 'eyes':
            # For eyes: >0.5 = Open (1), <=0.5 = Closed (0)
            predicted_labels = (predictions > 0.5).astype(int)
        else:
            # For yawn: >0.5 = Yawning (1), <=0.5 = Not yawning (0)
            predicted_labels = (predictions > 0.5).astype(int)
        
        true_labels = np.array(true_labels)
        
        # Calculate metrics
        self.analyze_results(true_labels, predicted_labels, predictions, image_paths, save_results)
    
    def analyze_results(self, true_labels, predicted_labels, raw_predictions, image_paths, save_results=True):
        """Detailed analysis of test results"""
        print(f"\nDETAILED RESULTS ANALYSIS")
        print("="*50)
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
        
        # AUC score
        try:
            auc_score = roc_auc_score(true_labels, raw_predictions)
        except:
            auc_score = 0.0
        
        print(f"Overall Performance:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   AUC Score: {auc_score:.4f}")
        
        # Per-class analysis
        print(f"\nPer-Class Performance:")
        class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
            true_labels, predicted_labels, average=None
        )
        
        for i, (prec, rec, f1_score, support) in enumerate(zip(class_precision, class_recall, class_f1, class_support)):
            class_name = self.get_label_meaning(i)
            print(f"   Class {i} - {class_name}:")
            print(f"     Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1_score:.4f}")
            print(f"     Support: {support} samples")
        
        # Confusion Matrix
        self.plot_confusion_matrix(true_labels, predicted_labels)
        
        # ROC Curve
        if auc_score > 0:
            self.plot_roc_curve(true_labels, raw_predictions)
        
        # Error analysis
        self.analyze_errors(true_labels, predicted_labels, raw_predictions, image_paths)
        
        # Save results
        if save_results:
            self.save_test_results(true_labels, predicted_labels, raw_predictions, image_paths)
    
    def plot_confusion_matrix(self, true_labels, predicted_labels):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predicted_labels)
        
        plt.figure(figsize=(10, 8))
        
        labels = [self.get_label_meaning(i) for i in range(2)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {self.detection_type.upper()} Detection', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add accuracy info
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.15, 0.02, f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                   fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, true_labels, raw_predictions):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(true_labels, raw_predictions)
        auc_score = roc_auc_score(true_labels, raw_predictions)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.detection_type.upper()} Detection', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def analyze_errors(self, true_labels, predicted_labels, raw_predictions, image_paths):
        """Analyze prediction errors"""
        print(f"\nERROR ANALYSIS")
        print("="*30)
        
        # Find misclassified samples
        misclassified = true_labels != predicted_labels
        num_errors = np.sum(misclassified)
        
        print(f"Total Errors: {num_errors} out of {len(true_labels)} ({num_errors/len(true_labels)*100:.2f}%)")
        
        if num_errors > 0:
            error_indices = np.where(misclassified)[0]
            
            # Analyze error types
            false_positives = np.where((true_labels == 0) & (predicted_labels == 1))[0]
            false_negatives = np.where((true_labels == 1) & (predicted_labels == 0))[0]
            
            if self.detection_type.lower() == 'eyes':
                print(f"False Positives (Predicted Open, Actually Closed): {len(false_positives)}")
                print(f"False Negatives (Predicted Closed, Actually Open): {len(false_negatives)}")
            else:
                print(f"False Positives (Predicted Yawn, Actually No Yawn): {len(false_positives)}")
                print(f"False Negatives (Predicted No Yawn, Actually Yawn): {len(false_negatives)}")
            
            # Show most confident wrong predictions
            error_confidences = []
            for idx in error_indices:
                pred_prob = raw_predictions[idx]
                confidence = max(pred_prob, 1 - pred_prob)
                error_confidences.append((idx, confidence, pred_prob))
            
            # Sort by confidence (most confident errors first)
            error_confidences.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nMost Confident Wrong Predictions (Top 5):")
            for i, (idx, confidence, pred_prob) in enumerate(error_confidences[:5]):
                true_label = true_labels[idx]
                pred_label = predicted_labels[idx]
                image_name = os.path.basename(image_paths[idx])
                
                true_class = self.get_label_meaning(true_label)
                pred_class = self.get_label_meaning(pred_label)
                
                print(f"   {i+1}. {image_name}")
                print(f"      True: {true_class} | Predicted: {pred_class}")
                print(f"      Confidence: {confidence:.4f} | Raw Score: {pred_prob:.4f}")
    
    def save_test_results(self, true_labels, predicted_labels, raw_predictions, image_paths):
        """Save detailed test results to files"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create results directory
            results_dir = f"test_results_{self.detection_type}_{timestamp}"
            os.makedirs(results_dir, exist_ok=True)
            
            # Save detailed results CSV
            results_df = pd.DataFrame({
                'image_path': image_paths,
                'image_name': [os.path.basename(p) for p in image_paths],
                'true_label': true_labels,
                'predicted_label': predicted_labels,
                'prediction_probability': raw_predictions,
                'correct_prediction': true_labels == predicted_labels,
                'confidence': np.maximum(raw_predictions, 1 - raw_predictions)
            })
            
            # Add human-readable labels
            results_df['true_class'] = results_df['true_label'].apply(self.get_label_meaning)
            results_df['predicted_class'] = results_df['predicted_label'].apply(self.get_label_meaning)
            
            csv_path = os.path.join(results_dir, 'detailed_results.csv')
            results_df.to_csv(csv_path, index=False)
            
            # Save summary report
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
            
            summary_path = os.path.join(results_dir, 'test_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write(f"DROWSINESS DETECTION MODEL TEST RESULTS\n")
                f.write("="*60 + "\n\n")
                f.write(f"Model: {self.model_path}\n")
                f.write(f"Detection Type: {self.detection_type.upper()}\n")
                f.write(f"Image Size: {self.img_size}x{self.img_size}\n")
                f.write(f"Test Date: {timestamp}\n\n")
                
                f.write("OVERALL PERFORMANCE:\n")
                f.write(f"Total Test Samples: {len(true_labels)}\n")
                f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1-Score: {f1:.4f}\n\n")
                
                # Per-class metrics
                class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
                    true_labels, predicted_labels, average=None
                )
                f.write("PER-CLASS PERFORMANCE:\n")
                for i, (prec, rec, f1_score, support) in enumerate(zip(class_precision, class_recall, class_f1, class_support)):
                    class_name = self.get_label_meaning(i)
                    f.write(f"Class {i} - {class_name}:\n")
                    f.write(f"  Precision: {prec:.4f}\n")
                    f.write(f"  Recall: {rec:.4f}\n")
                    f.write(f"  F1-Score: {f1_score:.4f}\n")
                    f.write(f"  Support: {support} samples\n\n")
            
            print(f"Test results saved to: {results_dir}/")
            print(f"   - Detailed results: detailed_results.csv")
            print(f"   - Summary report: test_summary.txt")
            
        except Exception as e:
            print(f"Could not save test results: {e}")
    
    def test_single_image(self, image_path, show_image=True):
        """Test a single image with detailed output"""
        if not self.model:
            print("No model loaded")
            return None
            
        print(f"\nTESTING SINGLE IMAGE")
        print("="*30)
        
        # Preprocess image
        img_batch = self.preprocess_image(image_path)
        if img_batch is None:
            return None
        
        # Make prediction
        try:
            raw_pred = self.model.predict(img_batch, verbose=0)[0][0]
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None
        
        # Interpret prediction based on detection type
        if self.detection_type.lower() == 'eyes':
            predicted_class = "Open Eyes (Alert)" if raw_pred > 0.5 else "Closed Eyes (Drowsy)"
            status = "ALERT" if raw_pred > 0.5 else "DROWSY"
        else:
            predicted_class = "Yawn (Drowsy)" if raw_pred > 0.5 else "No Yawn (Alert)"
            status = "DROWSY" if raw_pred > 0.5 else "ALERT"
        
        confidence = max(raw_pred, 1 - raw_pred)
        
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Prediction: {predicted_class}")
        print(f"Raw Score: {raw_pred:.4f}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
        print(f"Status: {status}")
        
        # Show image if requested
        if show_image:
            try:
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                plt.figure(figsize=(10, 8))
                plt.imshow(img_rgb)
                plt.title(f'{predicted_class}\nConfidence: {confidence:.3f} | Status: {status}', 
                         fontsize=14, weight='bold')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Could not display image: {e}")
        
        return {
            'raw_prediction': raw_pred,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'status': status
        }


def main():
    """Main function using existing label mapping from training"""
    print("DROWSINESS DETECTION MODEL COMPREHENSIVE TESTER - USING TRAINING LABELS")
    print("="*70)
    
    # ===== CONFIGURATION - UPDATE THESE PATHS =====
    MODEL_PATH = r"C:\Users\haniszaim\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\CODING\eyes_fixed_model_20250926_190534.h5"
    TEST_DIR = r'C:\Users\haniszaim\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\DATASET\DROWSINESS\EYES_CLOSED_OPENEDV4\test'
    
    # Model configuration
    IMG_SIZE = 96
    DETECTION_TYPE = 'eyes'
    BATCH_SIZE = 32
    # =============================================
    
    # Initialize tester
    print("Step 1: Initializing tester...")
    tester = DrowsinessModelTester(
        model_path=MODEL_PATH,
        img_size=IMG_SIZE,
        detection_type=DETECTION_TYPE
    )
    
    # Load model
    print("\nStep 2: Loading model...")
    if not tester.load_model():
        print("Failed to load model. Please check the model path.")
        return
    
    # Load training label mapping
    print("\nStep 3: Loading training label mapping...")
    mapping_file = f"label_mapping_{DETECTION_TYPE}.json"
    
    if tester.load_and_apply_training_labels(mapping_file):
        print("SUCCESS: Using exact same labels as training")
    else:
        print("WARNING: Using fallback label logic - results may be inconsistent")
    
    # Run comprehensive test
    print(f"\nStep 4: Running comprehensive test...")
    if os.path.exists(TEST_DIR):
        tester.comprehensive_test(TEST_DIR, batch_size=BATCH_SIZE, save_results=True)
    else:
        print(f"Test directory not found: {TEST_DIR}")
        return
    
    print(f"\nTo test individual images, use:")
    print(f"   result = tester.test_single_image('path/to/your/image.jpg')")
    
    print(f"\nTest completed successfully!")

if __name__ == "__main__":
    main()