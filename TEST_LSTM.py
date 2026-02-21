import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from tqdm import tqdm
import json
import time
import pickle
from concurrent.futures import ThreadPoolExecutor

class OptimizedSustDDDDataset(Dataset):
    """Same dataset class from your training script"""
    def __init__(self, data_dir, sequence_length=20, frame_size=(112, 112), 
                 transform=None, cache_frames=True, cache_dir=None):
        """
        Optimized Dataset class for SUST-DDD video data.
        
        Args:
            data_dir (str): Directory containing 'drowsy' and 'not_drowsy' folders
            sequence_length (int): Number of frames to extract from each video
            frame_size (tuple): Target size for frames
            transform: Optional data augmentation transforms
            cache_frames (bool): Whether to cache preprocessed frames
            cache_dir (str): Directory to store cached frames
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.transform = transform
        self.cache_frames = cache_frames
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.videos = []
        self.labels = []
        self.cached_data = {}
        
        # Load video paths and labels
        for class_name, label in [('drowsy', 1), ('not_drowsy', 0)]:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for video_path in class_dir.glob('*.mp4'):
                    self.videos.append(video_path)
                    self.labels.append(label)
        
        print(f"Loaded {len(self.videos)} videos from {data_dir}")
        print(f"  - Drowsy: {sum(self.labels)} videos")
        print(f"  - Not Drowsy: {len(self.labels) - sum(self.labels)} videos")
        
        # Preprocess and cache frames if enabled
        if self.cache_frames:
            self._preprocess_all_videos()
    
    def _preprocess_all_videos(self):
        """Preprocess all videos and cache them for faster loading."""
        print("Preprocessing and caching video frames...")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for idx, video_path in enumerate(self.videos):
                cache_file = self.cache_dir / f"{video_path.stem}_{self.sequence_length}_{self.frame_size[0]}.pkl"
                
                if not cache_file.exists():
                    future = executor.submit(self._process_and_cache_video, video_path, cache_file)
                    futures.append((idx, future))
                else:
                    # Load from cache
                    with open(cache_file, 'rb') as f:
                        self.cached_data[idx] = pickle.load(f)
            
            # Wait for all processing to complete
            for idx, future in tqdm(futures, desc="Processing videos"):
                self.cached_data[idx] = future.result()
    
    def _process_and_cache_video(self, video_path, cache_file):
        """Process a single video and cache the result."""
        frames = self.extract_frames_optimized(video_path)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(frames, f)
        
        return frames
    
    def extract_frames_optimized(self, video_path):
        """Optimized frame extraction using decord (fallback to OpenCV if not available)."""
        try:
            # Try using decord for faster video reading
            import decord
            video_reader = decord.VideoReader(str(video_path))
            total_frames = len(video_reader)
            
            if total_frames == 0:
                return np.zeros((self.sequence_length, 3, self.frame_size[1], self.frame_size[0]))
            
            # Sample frames evenly
            indices = np.linspace(0, max(0, total_frames - 1), self.sequence_length, dtype=int)
            frames = video_reader.get_batch(indices).asnumpy()
            
            # Resize and normalize
            processed_frames = []
            for frame in frames:
                frame = cv2.resize(frame, self.frame_size)
                frame = frame.astype(np.float32) / 255.0
                # Convert to CHW format
                frame = np.transpose(frame, (2, 0, 1))
                processed_frames.append(frame)
            
            return np.array(processed_frames)
            
        except ImportError:
            # Fallback to optimized OpenCV method
            return self.extract_frames_opencv_optimized(video_path)
    
    def extract_frames_opencv_optimized(self, video_path):
        """Optimized OpenCV frame extraction."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return np.zeros((self.sequence_length, 3, self.frame_size[1], self.frame_size[0]))
        
        # Sample frames evenly
        indices = np.linspace(0, max(0, total_frames - 1), self.sequence_length, dtype=int)
        
        # Read all frames at once for better performance
        all_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                # Convert to CHW format
                frame = np.transpose(frame, (2, 0, 1))
                all_frames.append(frame)
        
        cap.release()
        
        # Pad with last frame if needed
        while len(all_frames) < self.sequence_length:
            if all_frames:
                all_frames.append(all_frames[-1])
            else:
                all_frames.append(np.zeros((3, self.frame_size[1], self.frame_size[0])))
        
        return np.array(all_frames[:self.sequence_length])
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        # Get frames from cache if available, otherwise extract
        if idx in self.cached_data:
            frames = self.cached_data[idx]
        else:
            frames = self.extract_frames_optimized(self.videos[idx])
        
        # Apply transforms if any
        if self.transform:
            frames = self.transform(frames)
        
        # Convert to tensor
        frames = torch.from_numpy(frames)
        
        return frames, torch.tensor(label, dtype=torch.long)

class OptimizedLSTMDrowsinessDetector(nn.Module):
    """Same model class from your training script"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super(OptimizedLSTMDrowsinessDetector, self).__init__()
        
        # More efficient feature extractor using depthwise separable convolutions
        self.feature_extractor = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Depthwise separable conv block 1
            self._make_depthwise_block(32, 64, stride=2),
            self._make_depthwise_block(64, 128, stride=2),
            self._make_depthwise_block(128, 256, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.2)
        )
        
        # Calculate feature size after convolutions
        self.feature_size = 256 * 4 * 4
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_classes)
        )
    
    def _make_depthwise_block(self, in_channels, out_channels, stride=1):
        """Create depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise conv
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape to process all frames at once
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features for each frame
        features = self.feature_extractor(x)
        features = features.view(batch_size * seq_len, -1)
        
        # Reshape back to sequences
        features = features.view(batch_size, seq_len, -1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(features)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]
        
        # Classify
        output = self.classifier(last_output)
        
        return output

class DrowsinessDetectionTester:
    def __init__(self, model_path, device, results_dir="./test_results"):
        """
        Initialize the tester with trained model.
        
        Args:
            model_path (str): Path to the trained model checkpoint
            device: torch device (cuda or cpu)
            results_dir (str): Directory to save test results
        """
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Results will be saved to {results_dir}")
    
    def load_model(self, model_path):
        """Load the trained model from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model with same architecture
        model = OptimizedLSTMDrowsinessDetector(
            input_size=3 * 112 * 112,  # Assuming same config as training
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        # Print model info
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def test_model(self, test_loader, save_predictions=True, detailed_analysis=True):
        """
        Test the model on test dataset.
        
        Args:
            test_loader: DataLoader for test data
            save_predictions: Whether to save individual predictions
            detailed_analysis: Whether to perform detailed analysis
        """
        print("Starting model evaluation on test set...")
        print("-" * 60)
        
        start_time = time.time()
        
        # Initialize lists to store results
        all_predictions = []
        all_labels = []
        all_confidences = []
        video_names = []
        prediction_details = []
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Testing")
            
            for batch_idx, (videos, labels) in enumerate(progress_bar):
                videos = videos.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                with autocast():
                    outputs = self.model(videos)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidences, predicted = torch.max(probabilities, 1)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                
                # Store detailed predictions if requested
                if save_predictions:
                    batch_probs = probabilities.cpu().numpy()
                    for i in range(len(labels)):
                        prediction_details.append({
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'true_label': int(labels[i].cpu().item()),
                            'predicted_label': int(predicted[i].cpu().item()),
                            'confidence': float(confidences[i].cpu().item()),
                            'prob_not_drowsy': float(batch_probs[i][0]),
                            'prob_drowsy': float(batch_probs[i][1])
                        })
                
                # Update progress bar
                batch_acc = (predicted == labels).float().mean().item()
                progress_bar.set_postfix({
                    'Batch_Acc': f'{batch_acc:.3f}',
                    'Avg_Conf': f'{confidences.mean():.3f}'
                })
        
        test_time = time.time() - start_time
        
        # Calculate comprehensive metrics
        results = self.calculate_metrics(all_labels, all_predictions, all_confidences)
        results['test_time_seconds'] = test_time
        results['test_time_minutes'] = test_time / 60
        results['total_samples'] = len(all_labels)
        
        # Print results
        self.print_results(results)
        
        # Create visualizations
        self.create_visualizations(all_labels, all_predictions, all_confidences, results)
        
        # Save detailed results
        if save_predictions:
            self.save_detailed_predictions(prediction_details)
        
        # Perform detailed analysis if requested
        if detailed_analysis:
            self.detailed_analysis(all_labels, all_predictions, all_confidences)
        
        # Save summary results
        self.save_summary_results(results)
        
        return results
    
    def calculate_metrics(self, true_labels, predictions, confidences):
        """Calculate comprehensive evaluation metrics."""
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)
        
        # Confidence statistics
        avg_confidence = np.mean(confidences)
        confidence_correct = np.mean([confidences[i] for i in range(len(confidences)) 
                                    if predictions[i] == true_labels[i]])
        confidence_incorrect = np.mean([confidences[i] for i in range(len(confidences)) 
                                     if predictions[i] != true_labels[i]])
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        results = {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'avg_confidence': float(avg_confidence),
            'confidence_correct_predictions': float(confidence_correct),
            'confidence_incorrect_predictions': float(confidence_incorrect) if not np.isnan(confidence_incorrect) else 0.0,
            'class_names': ['Not Drowsy', 'Drowsy']
        }
        
        return results
    
    def print_results(self, results):
        """Print comprehensive test results."""
        print("\n" + "="*80)
        print("                        TEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Accuracy:           {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"   Precision (weighted): {results['precision_weighted']:.4f}")
        print(f"   Recall (weighted):    {results['recall_weighted']:.4f}")
        print(f"   F1-Score (weighted):  {results['f1_weighted']:.4f}")
        
        print(f"\n📈 PER-CLASS PERFORMANCE:")
        class_names = results['class_names']
        for i, class_name in enumerate(class_names):
            print(f"   {class_name}:")
            print(f"      Precision: {results['precision_per_class'][i]:.4f}")
            print(f"      Recall:    {results['recall_per_class'][i]:.4f}")
            print(f"      F1-Score:  {results['f1_per_class'][i]:.4f}")
        
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"   Average Confidence:              {results['avg_confidence']:.4f}")
        print(f"   Confidence (Correct Predictions): {results['confidence_correct_predictions']:.4f}")
        print(f"   Confidence (Incorrect Predictions): {results['confidence_incorrect_predictions']:.4f}")
        
        print(f"\n⏱️  TIMING:")
        print(f"   Test Time:          {results['test_time_minutes']:.2f} minutes")
        print(f"   Total Samples:      {results['total_samples']}")
        print(f"   Speed:              {results['total_samples']/results['test_time_seconds']:.2f} samples/second")
        
        print("\n" + "="*80)
    
    def create_visualizations(self, true_labels, predictions, confidences, results):
        """Create comprehensive visualizations."""
        # Set up the plot style
        plt.style.use('default')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=results['class_names'],
                   yticklabels=results['class_names'],
                   ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # 2. Per-class Performance
        ax2 = plt.subplot(2, 3, 2)
        metrics = ['Precision', 'Recall', 'F1-Score']
        not_drowsy_scores = [results['precision_per_class'][0], 
                            results['recall_per_class'][0], 
                            results['f1_per_class'][0]]
        drowsy_scores = [results['precision_per_class'][1], 
                        results['recall_per_class'][1], 
                        results['f1_per_class'][1]]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax2.bar(x - width/2, not_drowsy_scores, width, label='Not Drowsy', alpha=0.8)
        ax2.bar(x + width/2, drowsy_scores, width, label='Drowsy', alpha=0.8)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Per-Class Performance', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # 3. Confidence Distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(results['avg_confidence'], color='red', linestyle='--', 
                   label=f'Mean: {results["avg_confidence"]:.3f}')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        
        # 4. Confidence by Correctness
        ax4 = plt.subplot(2, 3, 4)
        correct_confidences = [confidences[i] for i in range(len(confidences)) 
                             if predictions[i] == true_labels[i]]
        incorrect_confidences = [confidences[i] for i in range(len(confidences)) 
                               if predictions[i] != true_labels[i]]
        
        ax4.boxplot([correct_confidences, incorrect_confidences], 
                   labels=['Correct', 'Incorrect'])
        ax4.set_ylabel('Confidence Score')
        ax4.set_title('Confidence by Prediction Correctness', fontsize=14, fontweight='bold')
        
        # 5. Class Distribution
        ax5 = plt.subplot(2, 3, 5)
        true_counts = [true_labels.count(0), true_labels.count(1)]
        pred_counts = [predictions.count(0), predictions.count(1)]
        
        x = np.arange(len(results['class_names']))
        width = 0.35
        
        ax5.bar(x - width/2, true_counts, width, label='True Labels', alpha=0.8)
        ax5.bar(x + width/2, pred_counts, width, label='Predictions', alpha=0.8)
        ax5.set_xlabel('Class')
        ax5.set_ylabel('Count')
        ax5.set_title('Class Distribution Comparison', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(results['class_names'])
        ax5.legend()
        
        # 6. Performance Summary
        ax6 = plt.subplot(2, 3, 6)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        scores = [results['accuracy'], results['precision_weighted'], 
                 results['recall_weighted'], results['f1_weighted']]
        
        bars = ax6.bar(metrics, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
        ax6.set_ylabel('Score')
        ax6.set_title('Overall Performance Summary', fontsize=14, fontweight='bold')
        ax6.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comprehensive_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to {self.results_dir / 'comprehensive_test_results.png'}")
    
    def save_detailed_predictions(self, prediction_details):
        """Save detailed prediction results."""
        with open(self.results_dir / 'detailed_predictions.json', 'w') as f:
            json.dump(prediction_details, f, indent=2)
        
        print(f"Detailed predictions saved to {self.results_dir / 'detailed_predictions.json'}")
    
    def detailed_analysis(self, true_labels, predictions, confidences):
        """Perform detailed analysis of results."""
        print("\n🔍 DETAILED ANALYSIS:")
        print("-" * 40)
        
        # Misclassification analysis
        misclassified = [(i, true_labels[i], predictions[i], confidences[i]) 
                        for i in range(len(true_labels)) 
                        if true_labels[i] != predictions[i]]
        
        print(f"Total Misclassifications: {len(misclassified)}")
        
        if misclassified:
            # Misclassification by type
            false_positives = [(i, conf) for i, true, pred, conf in misclassified if true == 0 and pred == 1]
            false_negatives = [(i, conf) for i, true, pred, conf in misclassified if true == 1 and pred == 0]
            
            print(f"False Positives (Not Drowsy → Drowsy): {len(false_positives)}")
            print(f"False Negatives (Drowsy → Not Drowsy): {len(false_negatives)}")
            
            # Most confident wrong predictions
            misclassified_sorted = sorted(misclassified, key=lambda x: x[3], reverse=True)
            print("\nTop 5 Most Confident Wrong Predictions:")
            for i, (idx, true, pred, conf) in enumerate(misclassified_sorted[:5]):
                true_label = "Drowsy" if true == 1 else "Not Drowsy"
                pred_label = "Drowsy" if pred == 1 else "Not Drowsy"
                print(f"  {i+1}. Sample {idx}: {true_label} → {pred_label} (Confidence: {conf:.4f})")
        
        # Confidence thresholds analysis
        print("\nConfidence Threshold Analysis:")
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        for threshold in thresholds:
            high_conf_indices = [i for i, conf in enumerate(confidences) if conf >= threshold]
            if high_conf_indices:
                high_conf_acc = sum(1 for i in high_conf_indices 
                                  if predictions[i] == true_labels[i]) / len(high_conf_indices)
                coverage = len(high_conf_indices) / len(confidences)
                print(f"  Threshold {threshold}: Accuracy = {high_conf_acc:.4f}, Coverage = {coverage:.4f}")
    
    def save_summary_results(self, results):
        """Save summary results to JSON file."""
        # Save complete results
        with open(self.results_dir / 'test_results_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create a simplified report
        report = {
            'model_performance': {
                'accuracy': f"{results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)",
                'precision': f"{results['precision_weighted']:.4f}",
                'recall': f"{results['recall_weighted']:.4f}",
                'f1_score': f"{results['f1_weighted']:.4f}"
            },
            'per_class_performance': {
                'not_drowsy': {
                    'precision': f"{results['precision_per_class'][0]:.4f}",
                    'recall': f"{results['recall_per_class'][0]:.4f}",
                    'f1_score': f"{results['f1_per_class'][0]:.4f}"
                },
                'drowsy': {
                    'precision': f"{results['precision_per_class'][1]:.4f}",
                    'recall': f"{results['recall_per_class'][1]:.4f}",
                    'f1_score': f"{results['f1_per_class'][1]:.4f}"
                }
            },
            'confidence_analysis': {
                'average_confidence': f"{results['avg_confidence']:.4f}",
                'confidence_when_correct': f"{results['confidence_correct_predictions']:.4f}",
                'confidence_when_incorrect': f"{results['confidence_incorrect_predictions']:.4f}"
            },
            'test_info': {
                'total_samples': results['total_samples'],
                'test_duration': f"{results['test_time_minutes']:.2f} minutes",
                'processing_speed': f"{results['total_samples']/results['test_time_seconds']:.2f} samples/second"
            }
        }
        
        with open(self.results_dir / 'test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Results saved to:")
        print(f"   - Summary: {self.results_dir / 'test_results_summary.json'}")
        print(f"   - Report: {self.results_dir / 'test_report.json'}")
        print(f"   - Visualizations: {self.results_dir / 'comprehensive_test_results.png'}")
    
    def test_single_video(self, video_path, sequence_length=20, frame_size=(112, 112)):
        """
        Test the model on a single video file.
        
        Args:
            video_path (str): Path to the video file
            sequence_length (int): Number of frames to extract
            frame_size (tuple): Frame size for processing
        
        Returns:
            dict: Prediction results
        """
        print(f"Testing single video: {video_path}")
        
        # Create temporary dataset for single video
        class SingleVideoDataset(Dataset):
            def __init__(self, video_path, sequence_length, frame_size):
                self.video_path = Path(video_path)
                self.sequence_length = sequence_length
                self.frame_size = frame_size
                
                if not self.video_path.exists():
                    raise FileNotFoundError(f"Video file not found: {video_path}")
            
            def extract_frames_opencv(self, video_path):
                """Extract frames using OpenCV."""
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames == 0:
                    cap.release()
                    return np.zeros((self.sequence_length, 3, self.frame_size[1], self.frame_size[0]))
                
                # Sample frames evenly
                indices = np.linspace(0, max(0, total_frames - 1), self.sequence_length, dtype=int)
                
                all_frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                for i in range(total_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if i in indices:
                        frame = cv2.resize(frame, self.frame_size)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = frame.astype(np.float32) / 255.0
                        frame = np.transpose(frame, (2, 0, 1))
                        all_frames.append(frame)
                
                cap.release()
                
                # Pad with last frame if needed
                while len(all_frames) < self.sequence_length:
                    if all_frames:
                        all_frames.append(all_frames[-1])
                    else:
                        all_frames.append(np.zeros((3, self.frame_size[1], self.frame_size[0])))
                
                return np.array(all_frames[:self.sequence_length])
            
            def __len__(self):
                return 1
            
            def __getitem__(self, idx):
                frames = self.extract_frames_opencv(self.video_path)
                frames = torch.from_numpy(frames)
                return frames, torch.tensor(0, dtype=torch.long)  # Dummy label
        
        # Create dataset and dataloader
        dataset = SingleVideoDataset(video_path, sequence_length, frame_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            for videos, _ in dataloader:
                videos = videos.to(self.device)
                
                with autocast():
                    outputs = self.model(videos)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                
                # Prepare results
                result = {
                    'video_path': str(video_path),
                    'predicted_class': int(predicted.item()),
                    'predicted_label': 'Drowsy' if predicted.item() == 1 else 'Not Drowsy',
                    'confidence': float(confidence.item()),
                    'probabilities': {
                        'not_drowsy': float(probabilities[0][0].item()),
                        'drowsy': float(probabilities[0][1].item())
                    }
                }
                
                print(f"Prediction: {result['predicted_label']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Probabilities: Not Drowsy: {result['probabilities']['not_drowsy']:.4f}, "
                      f"Drowsy: {result['probabilities']['drowsy']:.4f}")
                
                return result

def create_test_dataloader(test_data_dir, batch_size=8, sequence_length=20, 
                          frame_size=(112, 112), num_workers=4, cache_frames=True):
    """Create DataLoader for test data."""
    test_dataset = OptimizedSustDDDDataset(
        data_dir=test_data_dir,
        sequence_length=sequence_length,
        frame_size=frame_size,
        cache_frames=cache_frames,
        cache_dir=Path(test_data_dir).parent / 'cache' / 'test'
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return test_dataloader

def main():
    """Main testing function."""
    # Test Configuration
    TEST_CONFIG = {
        'model_path': r"C:\Users\haniszaim\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\CODING\optimized_drowsiness_checkpoints\best_model.pth",  # Path to your trained model
        'test_data_dir': r"C:\Users\haniszaim\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\DATASET\DROWSINESS\DDD_VIDEO_SPLIT\test",
        'batch_size': 8,
        'sequence_length': 24,
        'frame_size': (144, 144),
        'num_workers': 4,
        'cache_frames': True,
        'results_dir': './test_results',
        'save_predictions': True,
        'detailed_analysis': True
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
    
    # Check if model file exists
    model_path = Path(TEST_CONFIG['model_path'])
    if not model_path.exists():
        print(f"❌ Error: Model file not found at {model_path}")
        print("Make sure you have trained the model first and the path is correct.")
        return
    
    # Check if test data directory exists
    test_data_path = Path(TEST_CONFIG['test_data_dir'])
    if not test_data_path.exists():
        print(f"❌ Error: Test data directory not found at {test_data_path}")
        print("Make sure the test data path is correct.")
        return
    
    try:
        # Initialize tester
        print("Initializing Drowsiness Detection Tester...")
        tester = DrowsinessDetectionTester(
            model_path=TEST_CONFIG['model_path'],
            device=device,
            results_dir=TEST_CONFIG['results_dir']
        )
        
        # Create test data loader
        print("Creating test data loader...")
        test_loader = create_test_dataloader(
            test_data_dir=TEST_CONFIG['test_data_dir'],
            batch_size=TEST_CONFIG['batch_size'],
            sequence_length=TEST_CONFIG['sequence_length'],
            frame_size=TEST_CONFIG['frame_size'],
            num_workers=TEST_CONFIG['num_workers'],
            cache_frames=TEST_CONFIG['cache_frames']
        )
        
        print(f"Test dataset loaded: {len(test_loader.dataset)} videos")
        
        # Run comprehensive test
        results = tester.test_model(
            test_loader=test_loader,
            save_predictions=TEST_CONFIG['save_predictions'],
            detailed_analysis=TEST_CONFIG['detailed_analysis']
        )
        
        print("\n✅ Testing completed successfully!")
        print(f"📁 All results saved to: {TEST_CONFIG['results_dir']}")
        
        # Optional: Test single video example
        print("\n" + "="*60)
        print("SINGLE VIDEO TEST EXAMPLE")
        print("="*60)
        
        # Find a sample video from test data for demonstration
        sample_video = None
        for class_dir in ['drowsy', 'not_drowsy']:
            class_path = test_data_path / class_dir
            if class_path.exists():
                videos = list(class_path.glob('*.mp4'))
                if videos:
                    sample_video = videos[0]
                    break
        
        if sample_video:
            print(f"Testing sample video: {sample_video.name}")
            single_result = tester.test_single_video(
                video_path=sample_video,
                sequence_length=TEST_CONFIG['sequence_length'],
                frame_size=TEST_CONFIG['frame_size']
            )
        else:
            print("No sample video found for single video test.")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()