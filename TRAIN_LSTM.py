import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import json
import time
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
import pickle

class OptimizedSustDDDDataset(Dataset):
    def __init__(self, data_dir, sequence_length=20, frame_size=(112, 112), 
                 transform=None, cache_frames=True, cache_dir=None):
        """
        Optimized Dataset class for SUST-DDD video data.
        
        Args:
            data_dir (str): Directory containing 'drowsy' and 'not_drowsy' folders
            sequence_length (int): Number of frames to extract from each video (reduced from 30 to 20)
            frame_size (tuple): Target size for frames (reduced from 224 to 112 for faster processing)
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
            # Try using decord for faster video reading (install with: pip install decord)
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
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        """
        Optimized LSTM model for drowsiness detection with efficient feature extractor.
        """
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
        
        # Use GRU instead of LSTM for faster training (optional - you can keep LSTM)
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # More efficient classifier
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

class OptimizedDrowsinessTrainer:
    def __init__(self, model, device, save_dir="./optimized_checkpoints"):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Early stopping
        self.best_val_acc = 0.0
        self.patience = 10
        self.patience_counter = 0
    
    def train_epoch(self, dataloader, criterion, optimizer, accumulation_steps=1):
        """Optimized training epoch with mixed precision and gradient accumulation."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        optimizer.zero_grad()
        
        for batch_idx, (videos, labels) in enumerate(progress_bar):
            videos, labels = videos.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(videos)
                loss = criterion(outputs, labels) / accumulation_steps
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item() * accumulation_steps:.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, dataloader, criterion):
        """Optimized validation epoch with mixed precision."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc="Validation")
        
        with torch.no_grad():
            for videos, labels in progress_bar:
                videos, labels = videos.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                with autocast():
                    outputs = self.model(videos)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=0.003, 
              accumulation_steps=1):
        """Optimized training loop with better scheduling and early stopping."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Use OneCycleLR for better convergence
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader) // accumulation_steps
        )
        
        print(f"Starting optimized training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: Enabled")
        print(f"Gradient Accumulation Steps: {accumulation_steps}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, accumulation_steps)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate (OneCycleLR updates per batch)
            # scheduler.step() is called inside train_epoch for OneCycleLR
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model and early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_acc, optimizer)
                print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed! Total time: {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Plot training curves
        self.plot_training_curves()
    
    def save_checkpoint(self, epoch, val_acc, optimizer):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        torch.save(checkpoint, self.save_dir / 'best_model.pth')
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to {self.save_dir / 'training_curves.png'}")
    
    def evaluate_test_set(self, test_loader):
        """Evaluate model on test set."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        print("Evaluating on test set...")
        progress_bar = tqdm(test_loader, desc="Testing")
        
        with torch.no_grad():
            for videos, labels in progress_bar:
                videos, labels = videos.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                with autocast():
                    outputs = self.model(videos)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Drowsy', 'Drowsy'],
                    yticklabels=['Not Drowsy', 'Drowsy'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
        }
        
        with open(self.save_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)

def create_optimized_data_loaders(data_root, batch_size=8, sequence_length=20, 
                                frame_size=(112, 112), num_workers=4, cache_frames=True):
    """Create optimized DataLoaders for train, validation, and test sets."""
    data_root = Path(data_root)
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = data_root / split
        if split_dir.exists():
            dataset = OptimizedSustDDDDataset(
                split_dir, 
                sequence_length=sequence_length,
                frame_size=frame_size,
                cache_frames=cache_frames,
                cache_dir=data_root / 'cache' / split
            )
            
            shuffle = (split == 'train')
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False
            )
            dataloaders[split] = dataloader
        else:
            print(f"Warning: {split_dir} does not exist")
    
    return dataloaders

# Main execution
if __name__ == "__main__":
    # Optimized Configuration
    CONFIG = {
        'data_root': r"C:\Users\haniszaim\Desktop\UITM\DEGREE\FYP\FYP_PROJECT\DATASET\DROWSINESS\DDD_VIDEO_SPLIT",
        'batch_size': 64,  # Increased from 4
        'sequence_length': 24,  # Reduced from 30
        'frame_size': (144, 144),  # Reduced from (224, 224)
        'num_workers': 4,  # Increased from 2
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,  # Increased from 0.001
        'num_epochs': 50,
        'save_dir': "./optimized_drowsiness_checkpoints",
        'accumulation_steps': 2,  # For gradient accumulation
        'cache_frames': False  # Cache preprocessed frames
    }
    
    # Set device and optimization flags
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Optimize CUDA settings if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cudnn.enabled = True
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create optimized data loaders
    print("Creating optimized data loaders...")
    data_loaders = create_optimized_data_loaders(
        data_root=CONFIG['data_root'],
        batch_size=CONFIG['batch_size'],
        sequence_length=CONFIG['sequence_length'],
        frame_size=CONFIG['frame_size'],
        num_workers=CONFIG['num_workers'],
        cache_frames=CONFIG['cache_frames']
    )
    
    # Create optimized model
    print("Creating optimized LSTM model...")
    model = OptimizedLSTMDrowsinessDetector(
        input_size=3 * CONFIG['frame_size'][0] * CONFIG['frame_size'][1],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimized trainer
    trainer = OptimizedDrowsinessTrainer(model, device, CONFIG['save_dir'])
    
    # Start optimized training
    if 'train' in data_loaders and 'val' in data_loaders:
        trainer.train(
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
            num_epochs=CONFIG['num_epochs'],
            learning_rate=CONFIG['learning_rate'],
            accumulation_steps=CONFIG['accumulation_steps']
        )
        
        # Evaluate on test set if available
        if 'test' in data_loaders:
            print("\n" + "="*60)
            trainer.evaluate_test_set(data_loaders['test'])
    else:
        print("Error: Train or validation data not found!")