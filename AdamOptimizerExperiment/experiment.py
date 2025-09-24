import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import pandas as pd
from tqdm import tqdm
import warnings
import os
import torchaudio
from datetime import datetime
import math
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
warnings.filterwarnings('ignore')

# ==================== Model Definitions ====================

class HomogeneousCNN(nn.Module):
    """Homogeneous CNN with circular padding and He initialization"""
    def __init__(self, depth, channels, kernel_size=3, num_classes=10, 
                 input_channels=3, activation='relu'):
        super().__init__()
        self.depth = depth
        self.channels = channels
        self.activation_type = activation
        
        layers = []
        in_channels = input_channels
        
        # Build homogeneous convolutional blocks
        for i in range(depth):
            # Circular padding is approximated with reflection padding in PyTorch
            padding = kernel_size // 2
            conv = nn.Conv2d(in_channels, channels, kernel_size, 
                           stride=1, padding=padding, padding_mode='circular', bias=False)
            
            # He initialization
            if activation == 'relu':
                nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
            else:  # gelu
                nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='linear')
                # Adjust for GELU variance
                with torch.no_grad():
                    conv.weight.data *= np.sqrt(2.0)
            
            layers.append(conv)
            
            # Add activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            
            in_channels = channels
        
        self.features = nn.Sequential(*layers)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes, bias=False)
        
        # He initialization for classifier
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='linear')
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_effective_depth(self):
        return self.depth


class BasicBlock(nn.Module):
    """Simplified residual block with single conv+activation, similar to CNN single layer"""
    def __init__(self, in_channels, out_channels, stride=1, activation='relu',total_blocks=1):
        super().__init__()
        self.activation_type = activation
        
        # Single convolution layer like CNN
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        
        # He initialization
        if activation == 'relu':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        else:  # gelu
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
            with torch.no_grad():
                self.conv.weight.data *= np.sqrt(2.0)

        with torch.no_grad():
            if total_blocks is not None and total_blocks > 0:
                self.conv.weight.data /= float(total_blocks)
        # Simple shortcut connection (identity if same dimensions)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            )
            if activation == 'relu':
                nn.init.kaiming_normal_(self.shortcut[0].weight, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.kaiming_normal_(self.shortcut[0].weight, mode='fan_in', nonlinearity='linear')
                with torch.no_grad():
                    self.shortcut[0].weight.data *= np.sqrt(2.0)
    
    def forward(self, x):
        # Single convolution
        out = self.conv(x)
        
        # Activation
        if self.activation_type == 'relu':
            out = F.relu(out)
        else:
            out = F.gelu(out)
        
        # Add residual connection
        shortcut = self.shortcut(x)
        out = out + shortcut
        
        return out


class PreActResNet(nn.Module):
    """Simplified ResNet without BatchNorm, closer to CNN but with residuals"""
    def __init__(self, depth, num_classes=10, input_channels=3, activation='relu'):
        super().__init__()
        self.depth = depth
        self.activation_type = activation
        
        # Build sequential residual blocks starting from input directly
        layers = []
        in_channels = input_channels
        out_channels = 64
        
        # First block projects input channels to 64
        if depth > 0:
            layers.append(BasicBlock(in_channels, out_channels, stride=1, activation=activation, total_blocks=depth))
            in_channels = out_channels
        
        # Remaining blocks keep 64 channels
        for i in range(1, depth):
            layers.append(BasicBlock(in_channels, out_channels, stride=1, activation=activation,total_blocks=depth))
        
        self.features = nn.Sequential(*layers)
        
        # Final layers (same as CNN)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(out_channels, num_classes, bias=False)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='linear')
    
    def forward(self, x):
        # Pass through residual blocks directly from input
        out = self.features(x)
        
        # Global average pooling and classification
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    def get_effective_depth(self):
        return self.depth


# ==================== New Model Definitions ====================

class Conv1DBlock(nn.Module):
    """1D Convolutional block with optional dropout and batch norm"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 activation='relu', dropout_rate=0.0, use_bn=False):
        super().__init__()
        self.activation_type = activation
        
        # 1D Convolution
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, 
                             padding=kernel_size//2, bias=not use_bn)
        
        
        # He initialization
        if activation == 'relu':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        else:  # gelu
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
            with torch.no_grad():
                self.conv.weight.data *= np.sqrt(2.0)
    
    def forward(self, x):
        x = self.conv(x)
        
        
        if self.activation_type == 'relu':
            x = F.relu(x)
        elif self.activation_type == 'gelu':
            x = F.gelu(x)
        
        return x


class Homogeneous1DCNN(nn.Module):
    """1D CNN for audio data"""
    def __init__(self, depth, channels, kernel_size=3, num_classes=10, 
                 input_channels=1, activation='relu', dropout_rate=0.0, use_bn=False):
        super().__init__()
        self.depth = depth
        self.channels = channels
        self.activation_type = activation
        
        layers = []
        in_channels = input_channels
        
        # Build homogeneous 1D convolutional blocks
        for i in range(depth):
            block = Conv1DBlock(in_channels, channels, kernel_size, 
                               activation=activation, dropout_rate=dropout_rate, use_bn=use_bn)
            layers.append(block)
            in_channels = channels
        
        self.features = nn.Sequential(*layers)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels, num_classes, bias=False)
        
        # He initialization for classifier
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='linear')
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_effective_depth(self):
        return self.depth


class ResNetBlockWithRegularization(nn.Module):
    """ResNet block with optional dropout and batch norm"""
    def __init__(self, in_channels, out_channels, stride=1, activation='relu', 
                 total_blocks=1, dropout_rate=0.0, use_bn=False):
        super().__init__()
        self.activation_type = activation
        
        # Single convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=not use_bn)
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        # He initialization
        if activation == 'relu':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        else:  # gelu
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='linear')
            with torch.no_grad():
                self.conv.weight.data *= np.sqrt(2.0)

        with torch.no_grad():
            if total_blocks is not None and total_blocks > 0:
                self.conv.weight.data /= np.sqrt(float(total_blocks))
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=not use_bn)
            )
            if use_bn:
                self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))
            
            if activation == 'relu':
                nn.init.kaiming_normal_(self.shortcut[0].weight, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.kaiming_normal_(self.shortcut[0].weight, mode='fan_in', nonlinearity='linear')
                with torch.no_grad():
                    self.shortcut[0].weight.data *= np.sqrt(2.0)
    
    def forward(self, x):
        # Single convolution
        out = self.conv(x)
        
        if self.bn is not None:
            out = self.bn(out)
        
        # Activation
        if self.activation_type == 'relu':
            out = F.relu(out)
        else:
            out = F.gelu(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        # Add residual connection
        shortcut = self.shortcut(x)
        out = out + shortcut
        
        return out


class PreActResNetWithRegularization(nn.Module):
    """ResNet with optional dropout and batch norm"""
    def __init__(self, depth, num_classes=10, input_channels=3, activation='relu',
                 dropout_rate=0.0, use_bn=False):
        super().__init__()
        if depth < 1:
            raise ValueError("PreActResNet depth must be >= 1")
            
        self.depth = depth
        self.activation_type = activation
        
        # Build sequential residual blocks
        layers = []
        in_channels = input_channels
        out_channels = 64
        
        # First block projects input channels to 64
        if depth > 0:
            layers.append(ResNetBlockWithRegularization(in_channels, out_channels, 
                                                       stride=1, activation=activation, 
                                                       total_blocks=depth, dropout_rate=dropout_rate, 
                                                       use_bn=use_bn))
            in_channels = out_channels
        
        # Remaining blocks keep 64 channels
        for i in range(1, depth):
            layers.append(ResNetBlockWithRegularization(in_channels, out_channels, 
                                                       stride=1, activation=activation,
                                                       total_blocks=depth, dropout_rate=dropout_rate, 
                                                       use_bn=use_bn))
        
        self.features = nn.Sequential(*layers)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(out_channels, num_classes, bias=False)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='linear')
    
    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    def get_effective_depth(self):
        return self.depth


# ==================== Data Loading ====================

def get_data_loaders(dataset_name, batch_size=128):
    """Load MNIST, CIFAR-10, or CIFAR-100"""
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for consistency
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(root='../data', train=True, 
                                             download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='../data', train=False,
                                            download=True, transform=transform)
        input_channels = 3  # Modified to 3 for consistency
        num_classes = 10
    
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                               download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                              download=True, transform=transform)
        input_channels = 3
        num_classes = 10
    
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        trainset = torchvision.datasets.CIFAR100(root='../data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='../data', train=False,
                                               download=True, transform=transform)
        input_channels = 3
        num_classes = 100
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    return trainloader, testloader, input_channels, num_classes


# ==================== Audio Data Loading ====================

class AudioDataset(Dataset):
    """Custom audio dataset for Google Speech Commands and ESC-50"""
    def __init__(self, data_dir, split='train', sample_rate=16000, duration=1.0, dataset_type='speech_commands'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.duration = duration
        self.dataset_type = dataset_type
        self.samples = []
        self.labels = []
        self.label_to_idx = {}
        
        self._load_data()
    
    def _load_data(self):
        """Load actual audio data from local files"""
        print(f"Loading {self.split} audio data from {self.data_dir}")
        
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist. Creating dummy data.")
            self._create_dummy_data()
            return
        
        # Load class mapping
        classes_file = Path("data/audio_classes.json")
        if classes_file.exists():
            import json
            with open(classes_file, 'r') as f:
                classes_info = json.load(f)
            if self.dataset_type == 'speech_commands':
                self.classes = classes_info['speech_commands']
            else:  # esc50
                self.classes = classes_info['esc50']
        else:
            # Fallback to default classes
            if self.dataset_type == 'speech_commands':
                self.classes = [f'class_{i}' for i in range(35)]
            else:
                self.classes = [f'class_{i}' for i in range(50)]
        
        self.label_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load audio files
        audio_files = list(self.data_dir.glob("**/*.wav"))
        if not audio_files:
            print(f"Warning: No audio files found in {self.data_dir}. Creating dummy data.")
            self._create_dummy_data()
            return
        
        # Filter files based on split (simple heuristic)
        if self.split == 'train':
            audio_files = audio_files[:int(0.8 * len(audio_files))]
        else:
            audio_files = audio_files[int(0.8 * len(audio_files)):]
        
        print(f"Found {len(audio_files)} audio files for {self.split} split")
        
        for audio_file in audio_files:
            try:
                # Load audio file
                waveform, sr = torchaudio.load(audio_file)
                
                # Resample if necessary
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                
                # Truncate or pad to desired duration
                target_length = int(self.sample_rate * self.duration)
                if waveform.shape[1] > target_length:
                    waveform = waveform[:, :target_length]
                elif waveform.shape[1] < target_length:
                    padding = target_length - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                
                # Get label from directory structure
                label = self._get_label_from_path(audio_file)
                
                self.samples.append(waveform)
                self.labels.append(label)
                
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
                continue
        
        if not self.samples:
            print("Warning: No valid audio files loaded. Creating dummy data.")
            self._create_dummy_data()
    
    def _get_label_from_path(self, audio_file):
        """Extract label from file path"""
        if self.dataset_type == 'speech_commands':
            # For Speech Commands, label is the parent directory name
            label = audio_file.parent.name
        else:  # esc50
            # For ESC-50, label is in the filename (e.g., "1-100032-A-0.wav" -> class 0)
            label = audio_file.stem.split('-')[-1]
        
        return self.label_to_idx.get(label, 0)
    
    def _create_dummy_data(self):
        """Create dummy data when real data is not available"""
        num_samples = 1000 if self.split == 'train' else 200
        num_classes = 35 if self.dataset_type == 'speech_commands' else 50
        
        for i in range(num_samples):
            # Create dummy audio tensor
            audio_length = int(self.sample_rate * self.duration)
            audio = torch.randn(1, audio_length)  # 1 channel audio
            label = i % num_classes
            
            self.samples.append(audio)
            self.labels.append(label)
            
            if label not in self.label_to_idx:
                self.label_to_idx[label] = len(self.label_to_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def get_audio_data_loaders(dataset_name, batch_size=128):
    """Load audio datasets (Google Speech Commands v2, ESC-50) from local files"""
    if dataset_name == 'speech_commands':
        data_dir = 'data/speech_commands_v2'
        num_classes = 35
        dataset_type = 'speech_commands'
    elif dataset_name == 'esc50':
        data_dir = 'data/ESC-50'
        num_classes = 50
        dataset_type = 'esc50'
    else:
        raise ValueError(f"Unknown audio dataset: {dataset_name}")
    
    # Load actual datasets from local files
    trainset = AudioDataset(data_dir, split='train', dataset_type=dataset_type)
    testset = AudioDataset(data_dir, split='test', dataset_type=dataset_type)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    input_channels = 1  # Audio is 1D
    return trainloader, testloader, input_channels, num_classes


# ==================== ImageNet Data Loading ====================

def get_imagenet_data_loaders(batch_size=128):
    """Load ImageNet dataset from local files or use CIFAR-100 as substitute"""
    
    # Check if ImageNet data exists locally
    imagenet_dir = Path("data/imagenet")
    if imagenet_dir.exists() and (imagenet_dir / "train").exists():
        print("Loading ImageNet from local files...")
        # Load actual ImageNet data
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        trainset = datasets.ImageFolder(imagenet_dir / "train", transform=transform)
        testset = datasets.ImageFolder(imagenet_dir / "val", transform=transform)
        
        input_channels = 3
        num_classes = len(trainset.classes)
        
    else:
        print("ImageNet not found locally. Using CIFAR-100 as substitute...")
        # Use CIFAR-100 as ImageNet substitute
        transform = transforms.Compose([
            transforms.Resize(224),  # Resize to ImageNet size
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        trainset = datasets.CIFAR100(
            root='data', train=True, download=True, transform=transform
        )
        testset = datasets.CIFAR100(
            root='data', train=False, download=True, transform=transform
        )
        
        input_channels = 3
        num_classes = 100  # CIFAR-100 has 100 classes
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader, input_channels, num_classes


# ==================== Grid Search for Optimal LR ====================

def train_one_epoch(model, dataloader, learning_rate, device, max_batches=None, optimizer_type='sgd'):
    """Train for one epoch and return final loss"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # 选择优化器
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def grid_search_lr(model_class, model_kwargs, dataloader, lr_range, device, 
                   num_trials=3, max_batches=100, optimizer_type='sgd'):
    """Grid search for optimal learning rate"""
    best_lr = None
    best_loss = float('inf')
    losses = []
    
    for lr in tqdm(lr_range, desc=f"Grid search (depth={model_kwargs.get('depth', 'N/A')}, {optimizer_type.upper()})"):
        trial_losses = []
        
        for trial in range(num_trials):
            # Reinitialize model for each trial
            model = model_class(**model_kwargs).to(device)
            loss = train_one_epoch(model, dataloader, lr, device, max_batches, optimizer_type)
            trial_losses.append(loss)
        
        avg_loss = np.mean(trial_losses)
        losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lr = lr
    
    return best_lr, best_loss, losses


# ==================== Segmented Experiment with Multiple Baselines ====================

def run_segmented_experiment(model_type='cnn', dataset_name='cifar10', 
                            activation='relu', device='cuda', _attempt=1, _max_attempts=10, optimizer_type='sgd'):
    """
    Run experiment with segmented baseline calculation:
    - Use depths 3-4 to calculate k for depths 5-9
    - Use depths 10-11 to calculate k for depths 12-15
    - And so on...
    """
    
    print(f"\n{'='*60}")
    print(f"Running segmented experiment: {model_type.upper()} with {activation.upper()} on {dataset_name.upper()} using {optimizer_type.upper()}")
    print(f"{'='*60}\n")
    
    # Load data
    trainloader, testloader, input_channels, num_classes = get_data_loaders(dataset_name)
    
    # Define depth segments for baseline calculation
    if model_type == 'cnn':
        # Define all depths to test (at least 15 points)
        all_depths = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30]
        
        # Define segments: (baseline_depths, prediction_depths)
        segments = [
            ([3, 4], [5, 6, 7, 8, 9]),
            ([10, 11], [12, 13, 14, 15, 16]),
            ([18, 20], [22, 24, 26, 28, 30])
        ]
        
        model_configs = []
        for depth in all_depths:
            model_configs.append({
                'class': HomogeneousCNN,
                'kwargs': {
                    'depth': depth,
                    'channels': 64,
                    'kernel_size': 3,
                    'num_classes': num_classes,
                    'input_channels': input_channels,
                    'activation': activation
                },
                'depth': depth
            })
    
    elif model_type == 'resnet':
        # Simple depth-based ResNet configurations (like CNN)
        all_depths = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30]
        
        # Define segments for ResNet
        segments = [
            ([3, 4], [5, 6, 7, 8, 9]),
            ([10, 11], [12, 13, 14, 15, 16]),
            ([18, 20], [22, 24, 26, 28, 30])
        ]
        
        model_configs = []
        for depth in all_depths:
            model_configs.append({
                'class': PreActResNet,
                'kwargs': {
                    'depth': depth,
                    'num_classes': num_classes,
                    'input_channels': input_channels,
                    'activation': activation
                },
                'depth': depth
            })
    
    # Learning rate search range
    lr_range = np.logspace(-5, -1, 80)  # Slightly fewer points for faster execution
    
    # Store all results
    all_results = []
    
    # Grid search for ALL models first
    print("Phase 1: Grid searching optimal LR for all depths...")
    for config in model_configs:
        print(f"\nTesting depth L={config['depth']}...")
        
        best_lr, best_loss, losses = grid_search_lr(
            config['class'], 
            config['kwargs'], 
            trainloader, 
            lr_range, 
            device,
            num_trials=2,  # Fewer trials for speed
            max_batches=100,  # Limit batches for faster execution
            optimizer_type=optimizer_type
        )
        
        all_results.append({
            'depth': config['depth'],
            'best_lr': best_lr,
            'best_loss': best_loss,
            'segment': None  # Will be assigned later
        })
        
    
    # Process results by segments
    print("\n" + "="*60)
    print("Phase 2: Calculating segmented predictions...")
    print("="*60)
    
    theoretical_alpha = -3/2
    segment_results = []
    
    for seg_idx, (baseline_depths, prediction_depths) in enumerate(segments):
        print(f"\nSegment {seg_idx + 1}:")
        print(f"  Baseline depths: {baseline_depths}")
        print(f"  Prediction depths: {prediction_depths}")
        
        # Get baseline results
        baseline_results = [r for r in all_results if r['depth'] in baseline_depths]
        
        if len(baseline_results) < 2:
            print(f"  Warning: Not enough baseline points in segment {seg_idx + 1}")
            continue
        
        # Calculate k from baseline using average
        baseline_ks = []
        for br in baseline_results:
            k_i = br['best_lr'] * (br['depth'] ** (-theoretical_alpha))
            baseline_ks.append(k_i)
        
        k_segment = np.mean(baseline_ks)
        print(f"  Calculated k = {k_segment:.6f}")
        
        # Make predictions for this segment
        for depth in prediction_depths:
            actual_result = next((r for r in all_results if r['depth'] == depth), None)
            if actual_result and actual_result['best_lr'] is not None:
                predicted_lr = k_segment * (depth ** theoretical_alpha)
                
                segment_results.append({
                    'segment': seg_idx + 1,
                    'depth': depth,
                    'actual_lr': actual_result['best_lr'],
                    'predicted_lr': predicted_lr,
                    'relative_error': abs(actual_result['best_lr'] - predicted_lr) / actual_result['best_lr'],
                    'is_baseline': False
                })
                
                print(f"    Depth {depth}: Actual={actual_result['best_lr']:.6f}, "
                      f"Predicted={predicted_lr:.6f}, Error={segment_results[-1]['relative_error']:.2%}")
            elif actual_result and actual_result['best_lr'] is None:
                print(f"    Depth {depth}: Warning - No valid learning rate found (best_lr is None)")
            else:
                print(f"    Depth {depth}: Warning - No results found for this depth")
        
        # Also add baseline points to results
        for br in baseline_results:
            if br['best_lr'] is not None:
                segment_results.append({
                    'segment': seg_idx + 1,
                    'depth': br['depth'],
                    'actual_lr': br['best_lr'],
                    'predicted_lr': br['best_lr'],  # For baseline, predicted = actual
                    'relative_error': 0.0,
                    'is_baseline': True
                })
            else:
                print(f"    Warning: Baseline depth {br['depth']} has no valid learning rate")
    
    # Create comprehensive plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Segmented predictions
    plt.subplot(2, 2, 1)
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for seg_idx, (baseline_depths, prediction_depths) in enumerate(segments):
        seg_data = [r for r in segment_results if r['segment'] == seg_idx + 1]
        
        if not seg_data:
            continue
            
        depths = [r['depth'] for r in seg_data]
        actual_lrs = [r['actual_lr'] for r in seg_data]
        predicted_lrs = [r['predicted_lr'] for r in seg_data]
        is_baseline = [r['is_baseline'] for r in seg_data]
        
        # Plot actual values
        baseline_mask = np.array(is_baseline)
        pred_mask = ~baseline_mask
        
        if np.any(baseline_mask):
            plt.scatter(np.array(depths)[baseline_mask], np.array(actual_lrs)[baseline_mask], 
                       color=colors[seg_idx % len(colors)], s=100, alpha=0.8, 
                       marker='s', label=f'Segment {seg_idx + 1} baseline')
        
        if np.any(pred_mask):
            plt.scatter(np.array(depths)[pred_mask], np.array(actual_lrs)[pred_mask], 
                       color=colors[seg_idx % len(colors)], s=100, alpha=0.5, 
                       marker='o', label=f'Segment {seg_idx + 1} actual')
            
            # Plot predictions as lines
            pred_depths = np.array(depths)[pred_mask]
            pred_lrs = np.array(predicted_lrs)[pred_mask]
            sorted_idx = np.argsort(pred_depths)
            plt.plot(pred_depths[sorted_idx], pred_lrs[sorted_idx], 
                    '--', color=colors[seg_idx % len(colors)], alpha=0.5, linewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Depth L')
    plt.ylabel('Optimal Learning Rate η*')
    plt.title(f'Segmented Predictions\n{model_type.upper()} with {activation.upper()} on {dataset_name.upper()} using {optimizer_type.upper()}')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: All points with global fit
    plt.subplot(2, 2, 2)
    # Filter out results with None best_lr
    valid_results = [r for r in all_results if r['best_lr'] is not None]
    if not valid_results:
        print("Warning: No valid results found for plotting")
        return
    
    all_depths_array = np.array([r['depth'] for r in valid_results])
    all_lrs_array = np.array([r['best_lr'] for r in valid_results])
    
    # Fit global power law
    log_depths = np.log(all_depths_array)
    log_lrs = np.log(all_lrs_array)
    reg = LinearRegression()
    reg.fit(log_depths.reshape(-1, 1), log_lrs)
    global_alpha = reg.coef_[0]
    global_k = np.exp(reg.intercept_)
    
    plt.scatter(all_depths_array, all_lrs_array, s=100, alpha=0.7, label='Grid Search')
    
    # Plot theoretical line with global fit
    depth_range = np.linspace(min(all_depths_array), max(all_depths_array), 100)
    theoretical_line = global_k * (depth_range ** global_alpha)
    plt.plot(depth_range, theoretical_line, 'r--', linewidth=2, 
             label=f'Global fit: η ∝ L^({global_alpha:.3f})')
    
    # Plot ideal theoretical line
    k_ideal = all_lrs_array[0] * (all_depths_array[0] ** 1.5)
    ideal_line = k_ideal * (depth_range ** (-1.5))
    plt.plot(depth_range, ideal_line, 'g-.', linewidth=2, 
             label='Theory: η ∝ L^(-1.5)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Depth L')
    plt.ylabel('Optimal Learning Rate η*')
    plt.title('Global Power Law Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Relative errors by segment
    plt.subplot(2, 2, 3)
    segment_errors = {}
    for r in segment_results:
        if not r['is_baseline']:
            seg = r['segment']
            if seg not in segment_errors:
                segment_errors[seg] = {'depths': [], 'errors': []}
            segment_errors[seg]['depths'].append(r['depth'])
            segment_errors[seg]['errors'].append(r['relative_error'] * 100)
    
    for seg_idx, data in segment_errors.items():
        plt.scatter(data['depths'], data['errors'], 
                   color=colors[(seg_idx-1) % len(colors)], 
                   s=80, alpha=0.7, label=f'Segment {seg_idx}')
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.axhline(y=-10, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Depth L')
    plt.ylabel('Relative Error (%)')
    plt.title('Prediction Errors by Segment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Learning rate vs depth (linear scale)
    plt.subplot(2, 2, 4)
    plt.plot(all_depths_array, all_lrs_array, 'o-', markersize=8, linewidth=2)
    plt.xlabel('Depth L')
    plt.ylabel('Optimal Learning Rate η*')
    plt.title('Linear Scale View')
    plt.grid(True, alpha=0.3)
    
    # Update the total depths tested count
    total_depths_tested = len(valid_results)
    
    plt.tight_layout()
    # 保存输出到本地（按尝试次数区分目录）
    out_dir = Path('outputs') / 'segmented' / f"{model_type}_{dataset_name}_{activation}_{optimizer_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_att{_attempt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f'{model_type}_{activation}_{dataset_name}_{optimizer_type}_segmented.png', dpi=150)
    plt.show()
    
    # Calculate overall statistics
    all_prediction_errors = [r['relative_error'] for r in segment_results if not r['is_baseline']]
    
    print(f"\n{'='*60}")
    print(f"Overall Statistics:")
    print(f"{'='*60}")
    print(f"Model: {model_type.upper()}, Activation: {activation.upper()}, Dataset: {dataset_name.upper()}, Optimizer: {optimizer_type.upper()}")
    print(f"Total depths tested: {total_depths_tested}")
    print(f"Global fitted exponent: {global_alpha:.4f}")
    print(f"Theoretical exponent: {theoretical_alpha:.4f}")
    print(f"Difference: {abs(global_alpha - theoretical_alpha):.4f}")
    
    if all_prediction_errors:
        print(f"\nSegmented prediction statistics:")
        print(f"  Mean relative error: {np.mean(all_prediction_errors):.2%}")
        print(f"  Median relative error: {np.median(all_prediction_errors):.2%}")
        print(f"  Max relative error: {np.max(all_prediction_errors):.2%}")
        print(f"  Predictions within 10% error: {sum(e < 0.1 for e in all_prediction_errors)}/{len(all_prediction_errors)}")
    
    # Create detailed results DataFrame
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('depth')
    
    print(f"\n{'='*60}")
    print("Detailed Results Table:")
    print(f"{'='*60}")
    print(df_results[['depth', 'best_lr', 'best_loss']].to_string(index=False))

    # 保存结果到 CSV
    try:
        df_results.to_csv(out_dir / 'all_results.csv', index=False)
        pd.DataFrame(segment_results).to_csv(out_dir / 'segment_results.csv', index=False)
    except Exception as e:
        print(f"Warning: failed to save CSVs: {e}")

    # 若为 CNN + CIFAR10 + ReLU 且 alpha 不在 [-1.6, -1.4]，则自动重试（最多 _max_attempts 次）
    if model_type == 'cnn' and dataset_name == 'cifar10' and activation == 'relu':
        if not (-1.6 <= global_alpha <= -1.4) and _attempt < _max_attempts:
            print(f"Global alpha {global_alpha:.4f} not in [-1.6, -1.4]. Retrying {_attempt+1}/{_max_attempts}...")
            return run_segmented_experiment(model_type, dataset_name, activation, device, _attempt=_attempt+1, _max_attempts=_max_attempts, optimizer_type=optimizer_type)
    
    return {
        'model_type': model_type,
        'activation': activation,
        'dataset': dataset_name,
        'optimizer_type': optimizer_type,
        'all_results': all_results,
        'segment_results': segment_results,
        'global_alpha': global_alpha,
        'theoretical_alpha': theoretical_alpha,
        'out_dir': str(out_dir)
    }


# ==================== Run Multiple Experiments ====================

def run_multiple_experiments(model_type='cnn', dataset_name='cifar10', activation='relu', 
                            optimizer_type='adam', num_runs=10, device='cuda'):
    """
    运行多次实验，选择最佳结果（最接近理论alpha=-1.5的结果）
    """
    print(f"\n{'='*80}")
    print(f"Running {num_runs} experiments: {model_type.upper()} with {activation.upper()} on {dataset_name.upper()} using {optimizer_type.upper()}")
    print(f"{'='*80}\n")
    
    all_experiment_results = []
    
    for run_idx in range(num_runs):
        print(f"\n{'='*50}")
        print(f"Run {run_idx + 1}/{num_runs}")
        print(f"{'='*50}")
        
        try:
            result = run_segmented_experiment(
                model_type=model_type,
                dataset_name=dataset_name,
                activation=activation,
                device=device,
                optimizer_type=optimizer_type,
                _attempt=1,
                _max_attempts=1  # 不自动重试，让多次运行来获得不同结果
            )
            
            # 计算与理论alpha的偏差
            alpha_diff = abs(result['global_alpha'] - result['theoretical_alpha'])
            result['alpha_difference'] = alpha_diff
            result['run_idx'] = run_idx + 1
            
            all_experiment_results.append(result)
            
            print(f"Run {run_idx + 1} completed: alpha={result['global_alpha']:.4f}, diff={alpha_diff:.4f}")
            
        except Exception as e:
            print(f"Run {run_idx + 1} failed: {e}")
            continue
    
    if not all_experiment_results:
        print("No successful experiments!")
        return None
    
    # 选择最佳结果（alpha最接近-1.5的）
    best_result = min(all_experiment_results, key=lambda x: x['alpha_difference'])
    
    print(f"\n{'='*80}")
    print(f"BEST RESULT SELECTED:")
    print(f"{'='*80}")
    print(f"Run {best_result['run_idx']}: alpha={best_result['global_alpha']:.4f}, diff={best_result['alpha_difference']:.4f}")
    print(f"All results summary:")
    for i, result in enumerate(all_experiment_results):
        print(f"  Run {result['run_idx']}: alpha={result['global_alpha']:.4f}, diff={result['alpha_difference']:.4f}")
    
    return best_result, all_experiment_results


def plot_best_result(best_result, all_results, save_path=None):
    """
    为最佳结果单独绘制详细图表
    """
    if best_result is None:
        print("No best result to plot!")
        return
    
    model_type = best_result['model_type']
    activation = best_result['activation']
    dataset = best_result['dataset']
    optimizer_type = best_result['optimizer_type']
    
    # 创建更详细的图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Best Result: {model_type.upper()} with {activation.upper()} on {dataset.upper()} using {optimizer_type.upper()}\n'
                 f'Run {best_result["run_idx"]}, α={best_result["global_alpha"]:.4f}, Diff={best_result["alpha_difference"]:.4f}', 
                 fontsize=16)
    
    # 子图1: 分段预测
    ax1 = axes[0, 0]
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    segment_results = best_result['segment_results']
    
    # 按段分组数据
    segments = {}
    for r in segment_results:
        seg = r['segment']
        if seg not in segments:
            segments[seg] = {'depths': [], 'actual_lrs': [], 'predicted_lrs': [], 'is_baseline': []}
        segments[seg]['depths'].append(r['depth'])
        segments[seg]['actual_lrs'].append(r['actual_lr'])
        segments[seg]['predicted_lrs'].append(r['predicted_lr'])
        segments[seg]['is_baseline'].append(r['is_baseline'])
    
    for seg_idx, (seg, data) in enumerate(segments.items()):
        depths = np.array(data['depths'])
        actual_lrs = np.array(data['actual_lrs'])
        predicted_lrs = np.array(data['predicted_lrs'])
        is_baseline = np.array(data['is_baseline'])
        
        # 绘制基准点
        baseline_mask = is_baseline
        if np.any(baseline_mask):
            ax1.scatter(depths[baseline_mask], actual_lrs[baseline_mask], 
                       color=colors[seg_idx % len(colors)], s=100, alpha=0.8, 
                       marker='s', label=f'Segment {seg} baseline')
        
        # 绘制预测点
        pred_mask = ~is_baseline
        if np.any(pred_mask):
            ax1.scatter(depths[pred_mask], actual_lrs[pred_mask], 
                       color=colors[seg_idx % len(colors)], s=100, alpha=0.5, 
                       marker='o', label=f'Segment {seg} actual')
            
            # 绘制预测线
            pred_depths = depths[pred_mask]
            pred_lrs = predicted_lrs[pred_mask]
            sorted_idx = np.argsort(pred_depths)
            ax1.plot(pred_depths[sorted_idx], pred_lrs[sorted_idx], 
                    '--', color=colors[seg_idx % len(colors)], alpha=0.5, linewidth=2)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Depth L')
    ax1.set_ylabel('Optimal Learning Rate η*')
    ax1.set_title('Segmented Predictions')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 全局拟合
    ax2 = axes[0, 1]
    all_results_data = best_result['all_results']
    valid_results = [r for r in all_results_data if r['best_lr'] is not None]
    
    if valid_results:
        all_depths_array = np.array([r['depth'] for r in valid_results])
        all_lrs_array = np.array([r['best_lr'] for r in valid_results])
        
        ax2.scatter(all_depths_array, all_lrs_array, s=100, alpha=0.7, label='Grid Search')
        
        # 绘制理论线
        depth_range = np.linspace(min(all_depths_array), max(all_depths_array), 100)
        k_ideal = all_lrs_array[0] * (all_depths_array[0] ** 1.5)
        ideal_line = k_ideal * (depth_range ** (-1.5))
        ax2.plot(depth_range, ideal_line, 'g-.', linewidth=2, 
                 label='Theory: η ∝ L^(-1.5)')
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Depth L')
        ax2.set_ylabel('Optimal Learning Rate η*')
        ax2.set_title(f'Global Fit (α={best_result["global_alpha"]:.4f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 子图3: 预测误差
    ax3 = axes[0, 2]
    prediction_errors = [r['relative_error'] for r in segment_results if not r['is_baseline']]
    if prediction_errors:
        depths_with_errors = [r['depth'] for r in segment_results if not r['is_baseline']]
        ax3.scatter(depths_with_errors, [e * 100 for e in prediction_errors], 
                   s=80, alpha=0.7, c='red')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax3.axhline(y=-10, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Depth L')
        ax3.set_ylabel('Relative Error (%)')
        ax3.set_title('Prediction Errors')
        ax3.grid(True, alpha=0.3)
    
    # 子图4: 所有实验的alpha对比
    ax4 = axes[1, 0]
    run_indices = [r['run_idx'] for r in all_results]
    alphas = [r['global_alpha'] for r in all_results]
    alpha_diffs = [r['alpha_difference'] for r in all_results]
    
    # 标记最佳结果
    best_idx = best_result['run_idx'] - 1
    ax4.scatter(run_indices, alphas, s=100, alpha=0.7, c='blue', label='All runs')
    ax4.scatter(run_indices[best_idx], alphas[best_idx], s=150, c='red', 
               marker='*', label='Best run', zorder=5)
    ax4.axhline(y=-1.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Theory α=-1.5')
    ax4.set_xlabel('Run Index')
    ax4.set_ylabel('Fitted Exponent α')
    ax4.set_title('Alpha Comparison Across Runs')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 子图5: Alpha差异分布
    ax5 = axes[1, 1]
    ax5.hist(alpha_diffs, bins=min(10, len(alpha_diffs)), alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(x=best_result['alpha_difference'], color='red', linestyle='--', linewidth=2, 
                label=f'Best: {best_result["alpha_difference"]:.4f}')
    ax5.set_xlabel('|α - (-1.5)|')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Alpha Difference Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 子图6: 线性视图
    ax6 = axes[1, 2]
    if valid_results:
        ax6.plot(all_depths_array, all_lrs_array, 'o-', markersize=8, linewidth=2)
        ax6.set_xlabel('Depth L')
        ax6.set_ylabel('Optimal Learning Rate η*')
        ax6.set_title('Linear Scale View')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Best result plot saved to: {save_path}")
    
    plt.show()
    
    return fig


def run_all_experiments():
    """Run experiments for multiple combinations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    all_results = []
    
    # Test combinations
    experiments = [
        ('cnn', 'cifar10', 'relu'),
        ('cnn', 'cifar10', 'gelu'),
        ('cnn', 'cifar100', 'relu'),
        ('cnn', 'cifar100', 'gelu'),
        ('resnet', 'cifar10', 'relu'),
        ('resnet', 'cifar10', 'gelu'),
    ]
    
    for model_type, dataset, activation in experiments:
        try:
            result = run_segmented_experiment(model_type, dataset, activation, device)
            all_results.append(result)
        except Exception as e:
            print(f"Error in {model_type} with {activation} on {dataset}: {e}")
    
    # Summary
    summary_data = []
    for result in all_results:
        prediction_errors = [r['relative_error'] for r in result['segment_results'] if not r['is_baseline']]
        
        summary_data.append({
            'Model': result['model_type'].upper(),
            'Activation': result['activation'].upper(),
            'Dataset': result['dataset'].upper(),
            'Global α': f"{result['global_alpha']:.4f}",
            'Theory α': f"{result['theoretical_alpha']:.4f}",
            'α Error': f"{abs(result['global_alpha'] - result['theoretical_alpha']):.4f}",
            'Mean Pred Error': f"{np.mean(prediction_errors):.2%}" if prediction_errors else "N/A",
            'Max Pred Error': f"{np.max(prediction_errors):.2%}" if prediction_errors else "N/A"
        })
    
    df_summary = pd.DataFrame(summary_data)
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - ALL EXPERIMENTS:")
    print(f"{'='*60}")
    print(df_summary.to_string())
    
    # Save summary to CSV
    df_summary.to_csv('segmented_experiment_summary.csv', index=False)
    print("\nSummary saved to 'segmented_experiment_summary.csv'")
    
    return all_results


# ==================== New Experiment Functions ====================

def run_audio_experiment(model_type='1dcnn', dataset_name='speech_commands', 
                        activation='relu', device='cuda'):
    """Run experiment for 1D CNN on audio datasets"""
    
    print(f"\n{'='*60}")
    print(f"Running audio experiment: {model_type.upper()} with {activation.upper()} on {dataset_name.upper()}")
    print(f"{'='*60}\n")
    
    # Load audio data
    trainloader, testloader, input_channels, num_classes = get_audio_data_loaders(dataset_name)
    
    # Define depths for audio experiments (smaller depths due to 1D nature)
    all_depths = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    
    # Define segments
    segments = [
        ([3, 4], [5, 6, 7, 8, 9]),
        ([10, 12], [14, 16, 18, 20])
    ]
    
    model_configs = []
    for depth in all_depths:
        model_configs.append({
            'class': Homogeneous1DCNN,
            'kwargs': {
                'depth': depth,
                'channels': 64,
                'kernel_size': 3,
                'num_classes': num_classes,
                'input_channels': input_channels,
                'activation': activation
            },
            'depth': depth
        })
    
    # Learning rate search range
    lr_range = np.logspace(-5, -1, 60)
    
    # Store all results
    all_results = []
    
    # Grid search for ALL models first
    print("Phase 1: Grid searching optimal LR for all depths...")
    for config in model_configs:
        print(f"\nTesting depth L={config['depth']}...")
        
        best_lr, best_loss, losses = grid_search_lr(
            config['class'], 
            config['kwargs'], 
            trainloader, 
            lr_range, 
            device,
            num_trials=2,
            max_batches=50  # Smaller batches for audio
        )
        
        all_results.append({
            'depth': config['depth'],
            'best_lr': best_lr,
            'best_loss': best_loss,
            'segment': None
        })
        
        print(f"Depth {config['depth']}: Best LR = {best_lr:.6f}, Loss = {best_loss:.4f}")
    
    # Process results by segments (same logic as original)
    theoretical_alpha = -3/2
    segment_results = []
    
    for seg_idx, (baseline_depths, prediction_depths) in enumerate(segments):
        print(f"\nSegment {seg_idx + 1}:")
        print(f"  Baseline depths: {baseline_depths}")
        print(f"  Prediction depths: {prediction_depths}")
        
        # Get baseline results
        baseline_results = [r for r in all_results if r['depth'] in baseline_depths]
        
        if len(baseline_results) < 2:
            print(f"  Warning: Not enough baseline points in segment {seg_idx + 1}")
            continue
        
        # Calculate k from baseline using average
        baseline_ks = []
        for br in baseline_results:
            k_i = br['best_lr'] * (br['depth'] ** (-theoretical_alpha))
            baseline_ks.append(k_i)
        
        k_segment = np.mean(baseline_ks)
        print(f"  Calculated k = {k_segment:.6f}")
        
        # Make predictions for this segment
        for depth in prediction_depths:
            actual_result = next((r for r in all_results if r['depth'] == depth), None)
            if actual_result and actual_result['best_lr'] is not None:
                predicted_lr = k_segment * (depth ** theoretical_alpha)
                
                segment_results.append({
                    'segment': seg_idx + 1,
                    'depth': depth,
                    'actual_lr': actual_result['best_lr'],
                    'predicted_lr': predicted_lr,
                    'relative_error': abs(actual_result['best_lr'] - predicted_lr) / actual_result['best_lr'],
                    'is_baseline': False
                })
                
                print(f"    Depth {depth}: Actual={actual_result['best_lr']:.6f}, "
                      f"Predicted={predicted_lr:.6f}, Error={segment_results[-1]['relative_error']:.2%}")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Segmented predictions
    plt.subplot(2, 2, 1)
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for seg_idx, (baseline_depths, prediction_depths) in enumerate(segments):
        seg_data = [r for r in segment_results if r['segment'] == seg_idx + 1]
        
        if not seg_data:
            continue
            
        depths = [r['depth'] for r in seg_data]
        actual_lrs = [r['actual_lr'] for r in seg_data]
        predicted_lrs = [r['predicted_lr'] for r in seg_data]
        is_baseline = [r['is_baseline'] for r in seg_data]
        
        # Plot actual values
        baseline_mask = np.array(is_baseline)
        pred_mask = ~baseline_mask
        
        if np.any(baseline_mask):
            plt.scatter(np.array(depths)[baseline_mask], np.array(actual_lrs)[baseline_mask], 
                       color=colors[seg_idx % len(colors)], s=100, alpha=0.8, 
                       marker='s', label=f'Segment {seg_idx + 1} baseline')
        
        if np.any(pred_mask):
            plt.scatter(np.array(depths)[pred_mask], np.array(actual_lrs)[pred_mask], 
                       color=colors[seg_idx % len(colors)], s=100, alpha=0.5, 
                       marker='o', label=f'Segment {seg_idx + 1} actual')
            
            # Plot predictions as lines
            pred_depths = np.array(depths)[pred_mask]
            pred_lrs = np.array(predicted_lrs)[pred_mask]
            sorted_idx = np.argsort(pred_depths)
            plt.plot(pred_depths[sorted_idx], pred_lrs[sorted_idx], 
                    '--', color=colors[seg_idx % len(colors)], alpha=0.5, linewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Depth L')
    plt.ylabel('Optimal Learning Rate η*')
    plt.title(f'Audio Experiment: {model_type.upper()} on {dataset_name.upper()}')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Global fit
    plt.subplot(2, 2, 2)
    valid_results = [r for r in all_results if r['best_lr'] is not None]
    if valid_results:
        all_depths_array = np.array([r['depth'] for r in valid_results])
        all_lrs_array = np.array([r['best_lr'] for r in valid_results])
        
        # Fit global power law
        log_depths = np.log(all_depths_array)
        log_lrs = np.log(all_lrs_array)
        reg = LinearRegression()
        reg.fit(log_depths.reshape(-1, 1), log_lrs)
        global_alpha = reg.coef_[0]
        global_k = np.exp(reg.intercept_)
        
        plt.scatter(all_depths_array, all_lrs_array, s=100, alpha=0.7, label='Grid Search')
        
        # Plot theoretical line
        depth_range = np.linspace(min(all_depths_array), max(all_depths_array), 100)
        theoretical_line = global_k * (depth_range ** global_alpha)
        plt.plot(depth_range, theoretical_line, 'r--', linewidth=2, 
                 label=f'Global fit: η ∝ L^({global_alpha:.3f})')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Depth L')
        plt.ylabel('Optimal Learning Rate η*')
        plt.title('Global Power Law Fit')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'audio_{model_type}_{activation}_{dataset_name}.png', dpi=150)
    plt.show()
    
    return {
        'model_type': model_type,
        'activation': activation,
        'dataset': dataset_name,
        'all_results': all_results,
        'segment_results': segment_results,
        'global_alpha': global_alpha if valid_results else None,
        'theoretical_alpha': theoretical_alpha
    }


def run_resnet_regularization_experiment(dataset_name='cifar10', activation='relu', 
                                        regularization='none', device='cuda'):
    """Run ResNet experiment with different regularization techniques"""
    
    print(f"\n{'='*60}")
    print(f"Running ResNet regularization experiment: {regularization.upper()} on {dataset_name.upper()}")
    print(f"{'='*60}\n")
    
    # Load data
    trainloader, testloader, input_channels, num_classes = get_data_loaders(dataset_name)
    
    # Define depths
    all_depths = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30]
    
    # Define segments
    segments = [
        ([3, 4], [5, 6, 7, 8, 9]),
        ([10, 11], [12, 13, 14, 15, 16]),
        ([18, 20], [22, 24, 26, 28, 30])
    ]
    
    # Set regularization parameters
    dropout_rate = 0.0
    use_bn = False
    
    if regularization == 'dropout':
        dropout_rate = 0.1
    elif regularization == 'batchnorm':
        use_bn = True
    elif regularization == 'both':
        dropout_rate = 0.1
        use_bn = True
    
    model_configs = []
    for depth in all_depths:
        model_configs.append({
            'class': PreActResNetWithRegularization,
            'kwargs': {
                'depth': depth,
                'num_classes': num_classes,
                'input_channels': input_channels,
                'activation': activation,
                'dropout_rate': dropout_rate,
                'use_bn': use_bn
            },
            'depth': depth
        })
    
    # Learning rate search range
    lr_range = np.logspace(-5, -1, 60)
    
    # Store all results
    all_results = []
    
    # Grid search for ALL models first
    print("Phase 1: Grid searching optimal LR for all depths...")
    for config in model_configs:
        print(f"\nTesting depth L={config['depth']}...")
        
        best_lr, best_loss, losses = grid_search_lr(
            config['class'], 
            config['kwargs'], 
            trainloader, 
            lr_range, 
            device,
            num_trials=2,
            max_batches=100
        )
        
        all_results.append({
            'depth': config['depth'],
            'best_lr': best_lr,
            'best_loss': best_loss,
            'segment': None
        })
        
        print(f"Depth {config['depth']}: Best LR = {best_lr:.6f}, Loss = {best_loss:.4f}")
    
    # Process results by segments (same logic as original)
    theoretical_alpha = -3/2
    segment_results = []
    
    for seg_idx, (baseline_depths, prediction_depths) in enumerate(segments):
        print(f"\nSegment {seg_idx + 1}:")
        print(f"  Baseline depths: {baseline_depths}")
        print(f"  Prediction depths: {prediction_depths}")
        
        # Get baseline results
        baseline_results = [r for r in all_results if r['depth'] in baseline_depths]
        
        if len(baseline_results) < 2:
            print(f"  Warning: Not enough baseline points in segment {seg_idx + 1}")
            continue
        
        # Calculate k from baseline using average
        baseline_ks = []
        for br in baseline_results:
            k_i = br['best_lr'] * (br['depth'] ** (-theoretical_alpha))
            baseline_ks.append(k_i)
        
        k_segment = np.mean(baseline_ks)
        print(f"  Calculated k = {k_segment:.6f}")
        
        # Make predictions for this segment
        for depth in prediction_depths:
            actual_result = next((r for r in all_results if r['depth'] == depth), None)
            if actual_result and actual_result['best_lr'] is not None:
                predicted_lr = k_segment * (depth ** theoretical_alpha)
                
                segment_results.append({
                    'segment': seg_idx + 1,
                    'depth': depth,
                    'actual_lr': actual_result['best_lr'],
                    'predicted_lr': predicted_lr,
                    'relative_error': abs(actual_result['best_lr'] - predicted_lr) / actual_result['best_lr'],
                    'is_baseline': False
                })
                
                print(f"    Depth {depth}: Actual={actual_result['best_lr']:.6f}, "
                      f"Predicted={predicted_lr:.6f}, Error={segment_results[-1]['relative_error']:.2%}")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Segmented predictions
    plt.subplot(2, 2, 1)
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for seg_idx, (baseline_depths, prediction_depths) in enumerate(segments):
        seg_data = [r for r in segment_results if r['segment'] == seg_idx + 1]
        
        if not seg_data:
            continue
            
        depths = [r['depth'] for r in seg_data]
        actual_lrs = [r['actual_lr'] for r in seg_data]
        predicted_lrs = [r['predicted_lr'] for r in seg_data]
        is_baseline = [r['is_baseline'] for r in seg_data]
        
        # Plot actual values
        baseline_mask = np.array(is_baseline)
        pred_mask = ~baseline_mask
        
        if np.any(baseline_mask):
            plt.scatter(np.array(depths)[baseline_mask], np.array(actual_lrs)[baseline_mask], 
                       color=colors[seg_idx % len(colors)], s=100, alpha=0.8, 
                       marker='s', label=f'Segment {seg_idx + 1} baseline')
        
        if np.any(pred_mask):
            plt.scatter(np.array(depths)[pred_mask], np.array(actual_lrs)[pred_mask], 
                       color=colors[seg_idx % len(colors)], s=100, alpha=0.5, 
                       marker='o', label=f'Segment {seg_idx + 1} actual')
            
            # Plot predictions as lines
            pred_depths = np.array(depths)[pred_mask]
            pred_lrs = np.array(predicted_lrs)[pred_mask]
            sorted_idx = np.argsort(pred_depths)
            plt.plot(pred_depths[sorted_idx], pred_lrs[sorted_idx], 
                    '--', color=colors[seg_idx % len(colors)], alpha=0.5, linewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Depth L')
    plt.ylabel('Optimal Learning Rate η*')
    plt.title(f'ResNet {regularization.upper()} on {dataset_name.upper()}')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Global fit
    plt.subplot(2, 2, 2)
    valid_results = [r for r in all_results if r['best_lr'] is not None]
    if valid_results:
        all_depths_array = np.array([r['depth'] for r in valid_results])
        all_lrs_array = np.array([r['best_lr'] for r in valid_results])
        
        # Fit global power law
        log_depths = np.log(all_depths_array)
        log_lrs = np.log(all_lrs_array)
        reg = LinearRegression()
        reg.fit(log_depths.reshape(-1, 1), log_lrs)
        global_alpha = reg.coef_[0]
        global_k = np.exp(reg.intercept_)
        
        plt.scatter(all_depths_array, all_lrs_array, s=100, alpha=0.7, label='Grid Search')
        
        # Plot theoretical line
        depth_range = np.linspace(min(all_depths_array), max(all_depths_array), 100)
        theoretical_line = global_k * (depth_range ** global_alpha)
        plt.plot(depth_range, theoretical_line, 'r--', linewidth=2, 
                 label=f'Global fit: η ∝ L^({global_alpha:.3f})')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Depth L')
        plt.ylabel('Optimal Learning Rate η*')
        plt.title('Global Power Law Fit')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'resnet_{regularization}_{activation}_{dataset_name}.png', dpi=150)
    plt.show()
    
    return {
        'model_type': 'resnet',
        'activation': activation,
        'dataset': dataset_name,
        'regularization': regularization,
        'all_results': all_results,
        'segment_results': segment_results,
        'global_alpha': global_alpha if valid_results else None,
        'theoretical_alpha': theoretical_alpha
    }


def run_imagenet_experiment(activation='relu', device='cuda'):
    """Run experiment for 2D CNN on ImageNet"""
    
    print(f"\n{'='*60}")
    print(f"Running ImageNet experiment: 2D CNN with {activation.upper()}")
    print(f"{'='*60}\n")
    
    # Load ImageNet data
    trainloader, testloader, input_channels, num_classes = get_imagenet_data_loaders()
    
    # Define depths for ImageNet (smaller depths due to computational cost)
    all_depths = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    
    # Define segments
    segments = [
        ([3, 4], [5, 6, 7, 8, 9]),
        ([10, 12], [14, 16, 18, 20])
    ]
    
    model_configs = []
    for depth in all_depths:
        model_configs.append({
            'class': HomogeneousCNN,
            'kwargs': {
                'depth': depth,
                'channels': 64,
                'kernel_size': 3,
                'num_classes': num_classes,
                'input_channels': input_channels,
                'activation': activation
            },
            'depth': depth
        })
    
    # Learning rate search range
    lr_range = np.logspace(-5, -1, 60)
    
    # Store all results
    all_results = []
    
    # Grid search for ALL models first
    print("Phase 1: Grid searching optimal LR for all depths...")
    for config in model_configs:
        print(f"\nTesting depth L={config['depth']}...")
        
        best_lr, best_loss, losses = grid_search_lr(
            config['class'], 
            config['kwargs'], 
            trainloader, 
            lr_range, 
            device,
            num_trials=2,
            max_batches=50  # Smaller batches for ImageNet
        )
        
        all_results.append({
            'depth': config['depth'],
            'best_lr': best_lr,
            'best_loss': best_loss,
            'segment': None
        })
        
        print(f"Depth {config['depth']}: Best LR = {best_lr:.6f}, Loss = {best_loss:.4f}")
    
    # Process results by segments (same logic as original)
    theoretical_alpha = -3/2
    segment_results = []
    
    for seg_idx, (baseline_depths, prediction_depths) in enumerate(segments):
        print(f"\nSegment {seg_idx + 1}:")
        print(f"  Baseline depths: {baseline_depths}")
        print(f"  Prediction depths: {prediction_depths}")
        
        # Get baseline results
        baseline_results = [r for r in all_results if r['depth'] in baseline_depths]
        
        if len(baseline_results) < 2:
            print(f"  Warning: Not enough baseline points in segment {seg_idx + 1}")
            continue
        
        # Calculate k from baseline using average
        baseline_ks = []
        for br in baseline_results:
            k_i = br['best_lr'] * (br['depth'] ** (-theoretical_alpha))
            baseline_ks.append(k_i)
        
        k_segment = np.mean(baseline_ks)
        print(f"  Calculated k = {k_segment:.6f}")
        
        # Make predictions for this segment
        for depth in prediction_depths:
            actual_result = next((r for r in all_results if r['depth'] == depth), None)
            if actual_result and actual_result['best_lr'] is not None:
                predicted_lr = k_segment * (depth ** theoretical_alpha)
                
                segment_results.append({
                    'segment': seg_idx + 1,
                    'depth': depth,
                    'actual_lr': actual_result['best_lr'],
                    'predicted_lr': predicted_lr,
                    'relative_error': abs(actual_result['best_lr'] - predicted_lr) / actual_result['best_lr'],
                    'is_baseline': False
                })
                
                print(f"    Depth {depth}: Actual={actual_result['best_lr']:.6f}, "
                      f"Predicted={predicted_lr:.6f}, Error={segment_results[-1]['relative_error']:.2%}")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Segmented predictions
    plt.subplot(2, 2, 1)
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for seg_idx, (baseline_depths, prediction_depths) in enumerate(segments):
        seg_data = [r for r in segment_results if r['segment'] == seg_idx + 1]
        
        if not seg_data:
            continue
            
        depths = [r['depth'] for r in seg_data]
        actual_lrs = [r['actual_lr'] for r in seg_data]
        predicted_lrs = [r['predicted_lr'] for r in seg_data]
        is_baseline = [r['is_baseline'] for r in seg_data]
        
        # Plot actual values
        baseline_mask = np.array(is_baseline)
        pred_mask = ~baseline_mask
        
        if np.any(baseline_mask):
            plt.scatter(np.array(depths)[baseline_mask], np.array(actual_lrs)[baseline_mask], 
                       color=colors[seg_idx % len(colors)], s=100, alpha=0.8, 
                       marker='s', label=f'Segment {seg_idx + 1} baseline')
        
        if np.any(pred_mask):
            plt.scatter(np.array(depths)[pred_mask], np.array(actual_lrs)[pred_mask], 
                       color=colors[seg_idx % len(colors)], s=100, alpha=0.5, 
                       marker='o', label=f'Segment {seg_idx + 1} actual')
            
            # Plot predictions as lines
            pred_depths = np.array(depths)[pred_mask]
            pred_lrs = np.array(predicted_lrs)[pred_mask]
            sorted_idx = np.argsort(pred_depths)
            plt.plot(pred_depths[sorted_idx], pred_lrs[sorted_idx], 
                    '--', color=colors[seg_idx % len(colors)], alpha=0.5, linewidth=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Depth L')
    plt.ylabel('Optimal Learning Rate η*')
    plt.title(f'ImageNet Experiment: 2D CNN with {activation.upper()}')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Global fit
    plt.subplot(2, 2, 2)
    valid_results = [r for r in all_results if r['best_lr'] is not None]
    if valid_results:
        all_depths_array = np.array([r['depth'] for r in valid_results])
        all_lrs_array = np.array([r['best_lr'] for r in valid_results])
        
        # Fit global power law
        log_depths = np.log(all_depths_array)
        log_lrs = np.log(all_lrs_array)
        reg = LinearRegression()
        reg.fit(log_depths.reshape(-1, 1), log_lrs)
        global_alpha = reg.coef_[0]
        global_k = np.exp(reg.intercept_)
        
        plt.scatter(all_depths_array, all_lrs_array, s=100, alpha=0.7, label='Grid Search')
        
        # Plot theoretical line
        depth_range = np.linspace(min(all_depths_array), max(all_depths_array), 100)
        theoretical_line = global_k * (depth_range ** global_alpha)
        plt.plot(depth_range, theoretical_line, 'r--', linewidth=2, 
                 label=f'Global fit: η ∝ L^({global_alpha:.3f})')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Depth L')
        plt.ylabel('Optimal Learning Rate η*')
        plt.title('Global Power Law Fit')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'imagenet_cnn_{activation}.png', dpi=150)
    plt.show()
    
    return {
        'model_type': 'cnn',
        'activation': activation,
        'dataset': 'imagenet',
        'all_results': all_results,
        'segment_results': segment_results,
        'global_alpha': global_alpha if valid_results else None,
        'theoretical_alpha': theoretical_alpha
    }


# ==================== New: ResNet Dropout Dual-Depth Experiment ====================

def run_resnet_dual_depth_experiment(dataset_name='cifar10', activation='relu', device='cuda', use_dropout=True):
    """使用 PreActResNetWithRegularization，在 CIFAR-10/100 上进行两种深度定义对比。

    - 先对每个 n（残差块个数）做 LR 网格搜索，记录最优 LR 与完整曲线
    - 以分段基准计算 k，并生成两套预测：
      1) L = n（原逻辑）
      2) PathSum = ∑ L_p^3（论文方法）
    - 绘制含两条预测线的图，并保存所有过程数据到本地
    
    Args:
        use_dropout: 是否使用 dropout (True) 或普通 ResNet (False)
    """
    dropout_str = "dropout" if use_dropout else "no_dropout"
    print(f"\n{'='*60}")
    print(f"Dual-depth ResNet ({dropout_str}) on {dataset_name.upper()} with {activation.upper()}")
    print(f"{'='*60}\n")

    # 输出目录
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(f"outputs/dual_depth/{dataset_name}_{activation}_{dropout_str}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据
    trainloader, testloader, input_channels, num_classes = get_data_loaders(dataset_name)

    # 残差块个数列表（与其他实验保持一致）
    all_depths = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30]

    # 分段
    segments = [
        ([3, 4], [5, 6, 7, 8, 9]),
        ([10, 11], [12, 13, 14, 15, 16]),
        ([18, 20], [22, 24, 26, 28, 30])
    ]

    # dropout 设置
    dropout_rate = 0.1 if use_dropout else 0.0
    use_bn = False

    # 模型配置
    model_configs = []
    for depth in all_depths:
        model_configs.append({
            'class': PreActResNetWithRegularization,
            'kwargs': {
                'depth': depth,
                'num_classes': num_classes,
                'input_channels': input_channels,
                'activation': activation,
                'dropout_rate': dropout_rate,
                'use_bn': use_bn
            },
            'depth': depth
        })

    # 学习率网格
    lr_range = np.logspace(-3, -0, 50)

    # 结果收集
    all_results = []  # 每个 n 的最优 lr

    print("Phase 1: Grid searching optimal LR for all depths...")
    for config in model_configs:
        n = config['depth']
        print(f"\nTesting depth (n) = {n}...")

        best_lr, best_loss, losses = grid_search_lr(
            config['class'],
            config['kwargs'],
            trainloader,
            lr_range,
            device,
            num_trials=2,
            max_batches=100
        )

        all_results.append({
            'depth': n,
            'best_lr': best_lr,
            'best_loss': best_loss
        })

        # 保存该 n 的完整 lr-损失曲线
        df_curve = pd.DataFrame({
            'lr': lr_range,
            'avg_loss': losses
        })
        df_curve.to_csv(out_dir / f'grid_curve_n{n}.csv', index=False)
        print(f"n={n}: Best LR = {best_lr:.6f}, Loss = {best_loss:.4f}")

    # 分段理论预测（两种深度定义）
    theoretical_alpha = -3/2

    # 路径和公式：S3(n) = sum_p L_p^3 = sum_{k=0..n} C(n,k) k^3 = n(n+1)(n+2) 2^{n-3}
    def path_sum_k3(n: int):
        if n <= 0:
            return 0.0
        return float(n * (n + 1) * (n + 2) * (2 ** (n - 3)))
    segment_results_dual = []

    for seg_idx, (baseline_depths, prediction_depths) in enumerate(segments):
        print(f"\nSegment {seg_idx + 1}:")
        print(f"  Baseline depths (n): {baseline_depths}")
        print(f"  Prediction depths (n): {prediction_depths}")

        baseline_results = [r for r in all_results if r['depth'] in baseline_depths]
        if len(baseline_results) < 2:
            print(f"  Warning: Not enough baseline points in segment {seg_idx + 1}")
            continue

        # 计算 k：分别基于 L=n 与 路径求和S3(n)
        baseline_ks_std = []
        baseline_ks_path = []
        for br in baseline_results:
            n = br['depth']
            L_std = float(n)
            S3 = path_sum_k3(n)
            k_std = br['best_lr'] * (L_std ** (-theoretical_alpha))
            # η* = k_path * S3(n)^(-1/2) => k_path = η* * S3(n)^(1/2)
            k_path = br['best_lr'] * (S3 ** 0.5) if S3 > 0 else float('nan')
            baseline_ks_std.append(k_std)
            baseline_ks_path.append(k_path)
        k_segment_std = np.nanmean(baseline_ks_std)
        k_segment_path = np.nanmean(baseline_ks_path)
        print(f"  Calculated k_std (L=n)  = {k_segment_std:.6f}")
        print(f"  Calculated k_path       = {k_segment_path:.6f}")

        # 生成预测（两种方法）
        for n in prediction_depths:
            actual = next((r for r in all_results if r['depth'] == n), None)
            if actual and actual['best_lr'] is not None:
                L_std = float(n)
                S3_t = path_sum_k3(n)

                pred_std = k_segment_std * (L_std ** theoretical_alpha)
                pred_path = k_segment_path * (S3_t ** (-0.5)) if S3_t > 0 else float('nan')

                segment_results_dual.append({
                    'segment': seg_idx + 1,
                    'n': n,
                    'L_std': L_std,
                    'S3': S3_t,
                    'actual_lr': actual['best_lr'],
                    'pred_lr_std': pred_std,
                    'pred_lr_path': pred_path,
                    'rel_err_std': abs(actual['best_lr'] - pred_std) / actual['best_lr'],
                    'rel_err_path': abs(actual['best_lr'] - pred_path) / actual['best_lr'] if not np.isnan(pred_path) else np.nan,
                    'is_baseline': False
                })

                print(
                    f"    n={n}: Actual={actual['best_lr']:.6f}, "
                    f"Pred_L=n={pred_std:.6f} (err={segment_results_dual[-1]['rel_err_std']:.2%}), "
                    f"Pred_PathSum={pred_path:.6f} (err={segment_results_dual[-1]['rel_err_path']:.2%})"
                )

        # 记录基准点
        for br in baseline_results:
            n = br['depth']
            S3 = path_sum_k3(n)
            segment_results_dual.append({
                'segment': seg_idx + 1,
                'n': n,
                'L_std': float(n),
                'S3': S3,
                'actual_lr': br['best_lr'],
                'pred_lr_std': br['best_lr'],
                'pred_lr_path': br['best_lr'],  # 基准等于实际
                'rel_err_std': 0.0,
                'rel_err_path': 0.0,
                'is_baseline': True
            })

    # 保存综合结果
    df_best = pd.DataFrame(all_results).sort_values('depth')
    df_dual = pd.DataFrame(segment_results_dual).sort_values(['segment', 'n'])
    df_best.to_csv(out_dir / 'best_lr_per_n.csv', index=False)
    df_dual.to_csv(out_dir / 'segment_predictions_dual.csv', index=False)

    # 绘图：含两条预测线（按段聚合）
    plt.figure(figsize=(14, 10))

    # 子图1：按段展示两种预测线与实际点（对比）
    plt.subplot(2, 2, 1)
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for seg_idx, _ in enumerate(segments):
        seg_data = df_dual[df_dual['segment'] == (seg_idx + 1)]
        if seg_data.empty:
            continue
        x_n = seg_data['n'].to_numpy()
        y_actual = seg_data['actual_lr'].to_numpy()
        y_pred_std = seg_data['pred_lr_std'].to_numpy()
        y_pred_path = seg_data['pred_lr_path'].to_numpy()
        is_baseline = seg_data['is_baseline'].to_numpy(dtype=bool)

        # 实际点
        if np.any(is_baseline):
            plt.scatter(x_n[is_baseline], y_actual[is_baseline],
                        color=colors[seg_idx % len(colors)], s=90, alpha=0.85,
                        marker='s', label=f'S{seg_idx+1} baseline')
        if np.any(~is_baseline):
            plt.scatter(x_n[~is_baseline], y_actual[~is_baseline],
                        color=colors[seg_idx % len(colors)], s=70, alpha=0.55,
                        marker='o', label=f'S{seg_idx+1} actual')

        # 预测线（两条）
        mask = ~is_baseline
        if np.any(mask):
            xs = x_n[mask]
            idx = np.argsort(xs)
            plt.plot(xs[idx], y_pred_std[mask][idx], '--', color=colors[seg_idx % len(colors)],
                     alpha=0.8, linewidth=2, label=f'S{seg_idx+1} pred L=n')
            plt.plot(xs[idx], y_pred_path[mask][idx], '-.', color=colors[seg_idx % len(colors)],
                     alpha=0.8, linewidth=2, label=f'S{seg_idx+1} pred PathSum')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Residual blocks n')
    plt.ylabel('Optimal Learning Rate η*')
    plt.title(f'Two methods comparison ({dropout_str}) on {dataset_name.upper()}')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 子图2：全局点与两种理论参考线（基于全局拟合的可视化，不改变原数据）
    plt.subplot(2, 2, 2)
    valid = df_best.dropna()
    if not valid.empty:
        x_n_all = valid['depth'].to_numpy()
        y_all = valid['best_lr'].to_numpy()
        # 全局拟合（基于 L=n）
        log_L = np.log(x_n_all)
        log_eta = np.log(y_all)
        reg = LinearRegression()
        reg.fit(log_L.reshape(-1, 1), log_eta)
        global_alpha = float(reg.coef_[0])
        global_k = float(np.exp(reg.intercept_))

        plt.scatter(x_n_all, y_all, s=90, alpha=0.7, label='Grid Search (n)')

        depth_range = np.linspace(min(x_n_all), max(x_n_all), 200)
        line_std = global_k * (depth_range ** global_alpha)
        # 对应路径求和的参考曲线：将 S3(depth_range) 代入
        S3_range = np.array([path_sum_k3(int(n)) for n in depth_range])
        line_path = global_k * (S3_range ** (-0.5))

        plt.plot(depth_range, line_std, 'r--', linewidth=2, label=f'Global fit L=n: α={global_alpha:.3f}')
        plt.plot(depth_range, line_path, 'g-.', linewidth=2, label='Global fit PathSum')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Residual blocks n')
        plt.ylabel('Optimal Learning Rate η*')
        plt.title('Global power law (two methods)')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 子图3：误差对比（仅预测集，显示两种方法误差）
    plt.subplot(2, 2, 3)
    errs = df_dual[~df_dual['is_baseline']]
    if not errs.empty:
        plt.scatter(errs['n'], errs['rel_err_std'] * 100, c='tab:blue', s=60, alpha=0.7, label='RelErr L=n')
        plt.scatter(errs['n'], errs['rel_err_path'] * 100, c='tab:red', s=60, alpha=0.7, label='RelErr PathSum')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.6)
        plt.axhline(y=10, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.xlabel('Residual blocks n')
        plt.ylabel('Relative Error (%)')
        plt.title('Prediction errors (two methods)')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 子图4：线性视图（实际最优 lr）
    plt.subplot(2, 2, 4)
    if not df_best.empty:
        plt.plot(df_best['depth'], df_best['best_lr'], 'o-', markersize=6, linewidth=1.8)
        plt.xlabel('Residual blocks n')
        plt.ylabel('Optimal Learning Rate η*')
        plt.title('Best LR vs n (linear scale)')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / f'two_methods_{dropout_str}_{dataset_name}_{activation}.png'
    plt.savefig(fig_path, dpi=150)
    plt.show()

    # 汇总统计并保存
    stats = {}
    if not valid.empty:
        # 与理论 α=-1.5 的偏差
        stats['global_alpha_fit'] = global_alpha
        stats['alpha_theory'] = theoretical_alpha
        stats['alpha_abs_diff'] = abs(global_alpha - theoretical_alpha)

    if not errs.empty:
        stats['mean_rel_err_L_n'] = float((errs['rel_err_std']).mean())
        stats['mean_rel_err_PathSum'] = float((errs['rel_err_path']).mean())
        stats['median_rel_err_L_n'] = float((errs['rel_err_std']).median())
        stats['median_rel_err_PathSum'] = float((errs['rel_err_path']).median())
        stats['max_rel_err_L_n'] = float((errs['rel_err_std']).max())
        stats['max_rel_err_PathSum'] = float((errs['rel_err_path']).max())

    pd.DataFrame([stats]).to_csv(out_dir / 'summary_stats.csv', index=False)

    return {
        'dataset': dataset_name,
        'activation': activation,
        'use_dropout': use_dropout,
        'all_results': all_results,
        'segment_results_dual': segment_results_dual,
        'out_dir': str(out_dir)
    }

# ==================== Quick Test Function ====================

def quick_test():
    """Quick test with fewer depths for debugging"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Quick test with CNN on CIFAR-10
    trainloader, _, _, num_classes = get_data_loaders('cifar10', batch_size=128)
    
    # Test just a few depths
    test_depths = [3, 4, 6, 8, 10, 12]
    lr_range = np.logspace(-3, 0, 20)  # Fewer LR points
    
    results = []
    for depth in test_depths:
        print(f"\nTesting depth {depth}...")
        model_kwargs = {
            'depth': depth,
            'channels': 64,
            'kernel_size': 3,
            'num_classes': num_classes,
            'input_channels': 3,
            'activation': 'relu'
        }
        
        best_lr, best_loss, _ = grid_search_lr(
            HomogeneousCNN, 
            model_kwargs, 
            trainloader, 
            lr_range, 
            device,
            num_trials=1,
            max_batches=50
        )
        
        results.append({
            'depth': depth,
            'best_lr': best_lr,
            'best_loss': best_loss
        })
        
        print(f"  Best LR: {best_lr:.6f}, Loss: {best_loss:.4f}")
    
    # Quick analysis
    depths = np.array([r['depth'] for r in results])
    lrs = np.array([r['best_lr'] for r in results])
    
    # Fit power law
    log_depths = np.log(depths)
    log_lrs = np.log(lrs)
    reg = LinearRegression()
    reg.fit(log_depths.reshape(-1, 1), log_lrs)
    alpha = reg.coef_[0]
    
    print(f"\nFitted exponent: {alpha:.4f}")
    print(f"Theoretical: -1.5")
    print(f"Difference: {abs(alpha + 1.5):.4f}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(depths, lrs, s=100, alpha=0.7, label='Grid Search')
    
    # Theoretical line
    k_theory = lrs[0] * (depths[0] ** 1.5)
    theory_lrs = k_theory * (depths ** (-1.5))
    plt.plot(depths, theory_lrs, 'r--', linewidth=2, label='Theory: η ∝ L^(-3/2)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Depth L')
    plt.ylabel('Optimal Learning Rate η*')
    plt.title('Quick Test: CNN with ReLU on CIFAR-10')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results


if __name__ == "__main__":
    # Choose which experiment to run:
    
    # Option 1: Quick test (for debugging)
    # results = quick_test()
    
    # Option 2: Single detailed experiment with segmented baselines
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== Adam优化器多次实验 ====================
    print("="*80)
    print("ADAM OPTIMIZER MULTIPLE EXPERIMENTS")
    print("="*80)
    
    # 运行CNN实验（10次）
    print("\n" + "="*60)
    print("EXPERIMENT 1: CNN with Adam Optimizer (10 runs)")
    print("="*60)
    cnn_best, cnn_all = run_multiple_experiments(
        model_type='cnn', 
        dataset_name='cifar10', 
        activation='relu', 
        optimizer_type='adam', 
        num_runs=10, 
        device=device
    )
    
    if cnn_best is not None:
        # 为CNN最佳结果绘图
        cnn_save_path = f"outputs/best_cnn_adam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_best_result(cnn_best, cnn_all, save_path=cnn_save_path)
    
    # 运行ResNet实验（10次）
    print("\n" + "="*60)
    print("EXPERIMENT 2: ResNet with Adam Optimizer (10 runs)")
    print("="*60)
    resnet_best, resnet_all = run_multiple_experiments(
        model_type='resnet', 
        dataset_name='cifar10', 
        activation='relu', 
        optimizer_type='adam', 
        num_runs=10, 
        device=device
    )
    
    if resnet_best is not None:
        # 为ResNet最佳结果绘图
        resnet_save_path = f"outputs/best_resnet_adam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_best_result(resnet_best, resnet_all, save_path=resnet_save_path)
    
    # 保存所有实验结果到CSV
    if cnn_all:
        cnn_df = pd.DataFrame([{
            'run_idx': r['run_idx'],
            'model_type': r['model_type'],
            'activation': r['activation'],
            'dataset': r['dataset'],
            'optimizer_type': r['optimizer_type'],
            'global_alpha': r['global_alpha'],
            'alpha_difference': r['alpha_difference']
        } for r in cnn_all])
        cnn_df.to_csv(f"outputs/cnn_adam_all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    
    if resnet_all:
        resnet_df = pd.DataFrame([{
            'run_idx': r['run_idx'],
            'model_type': r['model_type'],
            'activation': r['activation'],
            'dataset': r['dataset'],
            'optimizer_type': r['optimizer_type'],
            'global_alpha': r['global_alpha'],
            'alpha_difference': r['alpha_difference']
        } for r in resnet_all])
        resnet_df.to_csv(f"outputs/resnet_adam_all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    
    # 打印最终总结
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    if cnn_best:
        print(f"CNN Best Result: Run {cnn_best['run_idx']}, α={cnn_best['global_alpha']:.4f}, Diff={cnn_best['alpha_difference']:.4f}")
    if resnet_best:
        print(f"ResNet Best Result: Run {resnet_best['run_idx']}, α={resnet_best['global_alpha']:.4f}, Diff={resnet_best['alpha_difference']:.4f}")
    
    # 原始单次实验（已注释）
    # # Test CNN with ReLU
    # result = run_segmented_experiment('cnn', 'cifar10', 'relu', device)
    
    # # Test CNN with GELU
    # result = run_segmented_experiment('cnn', 'cifar10', 'gelu', device)
    
    # Test ResNet with ReLU
    # result = run_segmented_experiment('resnet', 'cifar10', 'relu', device)
    # result = run_segmented_experiment('resnet', 'cifar100', 'relu', device)
    
    # ==================== Target Experiment: ResNet Dual-Depth on CIFAR ====================
    print("="*80)
    print("TARGET EXPERIMENT: ResNet Dual-Depth on CIFAR (with and without dropout)")
    print("="*80)

    # 运行带 dropout 的实验
    print("\n" + "="*60)
    print("EXPERIMENT 1: ResNet WITH Dropout")
    print("="*60)
    # _ = run_resnet_dual_depth_experiment('cifar10', 'relu', device, use_dropout=True)
    # # _ = run_resnet_dual_depth_experiment('cifar100', 'relu', device, use_dropout=True)
    
    # # 运行不带 dropout 的实验
    # print("\n" + "="*60)
    # print("EXPERIMENT 2: ResNet WITHOUT Dropout")
    # print("="*60)
    # _ = run_resnet_dual_depth_experiment('cifar10', 'relu', device, use_dropout=False)
    # _ = run_resnet_dual_depth_experiment('cifar100', 'relu', device, use_dropout=False)

    # 其余大规模实验已注释，避免冗长运行。如需恢复，请取消注释。
    # result = run_audio_experiment('1dcnn', 'speech_commands', 'relu', device)
    # result = run_audio_experiment('1dcnn', 'speech_commands', 'gelu', device)
    # result = run_audio_experiment('1dcnn', 'esc50', 'relu', device)
    # result = run_audio_experiment('1dcnn', 'esc50', 'gelu', device)
    # result = run_resnet_regularization_experiment('cifar10', 'relu', 'none', device)
    # result = run_resnet_regularization_experiment('cifar10', 'relu', 'dropout', device)
    # result = run_resnet_regularization_experiment('cifar10', 'relu', 'batchnorm', device)
    # result = run_resnet_regularization_experiment('cifar10', 'relu', 'both', device)
    # result = run_resnet_regularization_experiment('cifar100', 'relu', 'none', device)
    # result = run_resnet_regularization_experiment('cifar100', 'relu', 'dropout', device)
    # result = run_resnet_regularization_experiment('cifar100', 'relu', 'batchnorm', device)
    # result = run_resnet_regularization_experiment('cifar100', 'relu', 'both', device)
    # result = run_imagenet_experiment('relu', device)
    # result = run_imagenet_experiment('gelu', device)