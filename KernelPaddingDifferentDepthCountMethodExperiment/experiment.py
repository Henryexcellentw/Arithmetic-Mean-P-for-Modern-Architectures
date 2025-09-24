import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
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
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
warnings.filterwarnings('ignore')

# ==================== Model Definitions ====================

class HomogeneousCNN(nn.Module):
    """Homogeneous CNN with configurable padding and He initialization"""
    def __init__(self, depth, channels, kernel_size=3, num_classes=10, 
                 input_channels=3, activation='relu', padding_mode='circular'):
        super().__init__()
        self.depth = depth
        self.channels = channels
        self.activation_type = activation
        self.padding_mode = padding_mode
        
        layers = []
        in_channels = input_channels
        
        # Build homogeneous convolutional blocks
        for i in range(depth):
            # Circular padding is approximated with reflection padding in PyTorch
            padding = kernel_size // 2
            conv = nn.Conv2d(
                in_channels,
                channels,
                kernel_size,
                stride=1,
                padding=padding,
                padding_mode=self.padding_mode,
                bias=False,
            )
            
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
    imagenet_dir = Path("../data/imagenet")
    if imagenet_dir.exists() and (imagenet_dir / "train").exists():
        print("Loading ImageNet from local files...")
        # Load actual ImageNet data
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        trainset = torchvision.datasets.ImageFolder(imagenet_dir / "train", transform=transform)
        testset = torchvision.datasets.ImageFolder(imagenet_dir / "val", transform=transform)
        
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
        
        trainset = torchvision.datasets.CIFAR100(
            root='data', train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR100(
            root='data', train=False, download=True, transform=transform
        )
        
        input_channels = 3
        num_classes = 100  # CIFAR-100 has 100 classes
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader, input_channels, num_classes


# ==================== Grid Search for Optimal LR ====================

def train_one_epoch(model, dataloader, learning_rate, device, max_batches=None):
    """Train for one epoch and return final loss"""
    model.train()
    criterion = nn.CrossEntropyLoss()
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
                   num_trials=3, max_batches=100):
    """Grid search for optimal learning rate"""
    best_lr = None
    best_loss = float('inf')
    losses = []
    
    for lr in tqdm(lr_range, desc=f"Grid search (depth={model_kwargs.get('depth', 'N/A')})"):
        trial_losses = []
        
        for trial in range(num_trials):
            # Reinitialize model for each trial
            model = model_class(**model_kwargs).to(device)
            loss = train_one_epoch(model, dataloader, lr, device, max_batches)
            trial_losses.append(loss)
        
        avg_loss = np.mean(trial_losses)
        losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lr = lr
    
    return best_lr, best_loss, losses


# ==================== Segmented Experiment with Multiple Baselines ====================

def run_segmented_experiment(model_type='cnn', dataset_name='cifar10', 
                            activation='relu', device='cuda'):
    """
    Run experiment with segmented baseline calculation:
    - Use depths 3-4 to calculate k for depths 5-9
    - Use depths 10-11 to calculate k for depths 12-15
    - And so on...
    """
    
    print(f"\n{'='*60}")
    print(f"Running segmented experiment: {model_type.upper()} with {activation.upper()} on {dataset_name.upper()}")
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
        all_depths = [ ]
        
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
    lr_range = np.logspace(-3, 0, 50)   # Slightly fewer points for faster execution
    
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
            max_batches=None  # Limit batches for faster execution
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
    plt.title(f'Segmented Predictions\n{model_type.upper()} with {activation.upper()} on {dataset_name.upper()}')
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
    plt.savefig(f'{model_type}_{activation}_{dataset_name}_segmented.png', dpi=150)
    plt.show()
    
    # Calculate overall statistics
    all_prediction_errors = [r['relative_error'] for r in segment_results if not r['is_baseline']]
    
    print(f"\n{'='*60}")
    print(f"Overall Statistics:")
    print(f"{'='*60}")
    print(f"Model: {model_type.upper()}, Activation: {activation.upper()}, Dataset: {dataset_name.upper()}")
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
    
    return {
        'model_type': model_type,
        'activation': activation,
        'dataset': dataset_name,
        'all_results': all_results,
        'segment_results': segment_results,
        'global_alpha': global_alpha,
        'theoretical_alpha': theoretical_alpha
    }


# ==================== Run Multiple Experiments ====================

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
    lr_range = np.logspace(-3, 0, 50) 
    
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
    if valid_results:
        plt.plot(all_depths_array, all_lrs_array, 'o-', markersize=8, linewidth=2)
    plt.xlabel('Depth L')
    plt.ylabel('Optimal Learning Rate η*')
    plt.title('Linear Scale View')
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


# ==================== Requested Experiments (User) ====================

def run_cnn_kernel_size_experiment(dataset_name='cifar10', activation='relu', device='cuda'):
    """Compare fitted exponent α for CNN with kernel sizes 3/4/5 on CIFAR-10.
    Other settings are identical to baseline; padding keeps spatial size.
    """
    trainloader, _, input_channels, num_classes = get_data_loaders(dataset_name)

    depths = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30]
    lr_range = np.logspace(-3, 0, 50)

    kernel_list = [3, 4, 5]
    kernel_to_results = {}
    kernel_to_alpha = {}
    kernel_to_fit = {}

    for k in kernel_list:
        results = []
        for d in depths:
            model_kwargs = {
                'depth': d,
                'channels': 64,
                'kernel_size': k,
                'num_classes': num_classes,
                'input_channels': input_channels,
                'activation': activation,
                'padding_mode': 'circular',
            }
            best_lr, best_loss, _ = grid_search_lr(HomogeneousCNN, model_kwargs, trainloader, lr_range, device, num_trials=2, max_batches=None)
            results.append({'depth': d, 'best_lr': best_lr, 'best_loss': best_loss})
        kernel_to_results[k] = results

        d_arr = np.array([r['depth'] for r in results])
        lr_arr = np.array([r['best_lr'] for r in results])
        reg = LinearRegression().fit(np.log(d_arr).reshape(-1, 1), np.log(lr_arr))
        alpha = reg.coef_[0]
        kernel_to_alpha[k] = alpha
        kernel_to_fit[k] = (np.exp(reg.intercept_), alpha)

    # Save results
    os.makedirs('results', exist_ok=True)
    for k in kernel_list:
        pd.DataFrame(kernel_to_results[k]).to_csv(f'results/cnn_kernel_size_results_k{k}.csv', index=False)
    pd.DataFrame({
        'kernel': kernel_list,
        'alpha': [kernel_to_alpha[k] for k in kernel_list]
    }).to_csv('results/cnn_kernel_size_summary.csv', index=False)

    # Plot three fitted lines and scatters
    plt.figure(figsize=(8, 6))
    colors = {3: 'tab:blue', 4: 'tab:green', 5: 'tab:red'}
    for k in kernel_list:
        results = kernel_to_results[k]
        d_arr = np.array([r['depth'] for r in results])
        lr_arr = np.array([r['best_lr'] for r in results])
        plt.scatter(d_arr, lr_arr, s=60, alpha=0.7, color=colors[k], label=f'k={k} data')
        k_fit, a_fit = kernel_to_fit[k]
        d_grid = np.linspace(min(d_arr), max(d_arr), 100)
        plt.plot(d_grid, k_fit * (d_grid ** a_fit), '--', color=colors[k], linewidth=2, label=f'k={k} fit: α={a_fit:.3f}')

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Depth L'); plt.ylabel('Optimal LR η*')
    plt.title(f'CNN kernel size comparison on {dataset_name.upper()} ({activation.upper()})')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('cnn_kernel_size_compare.png', dpi=150); plt.show()

    print('Fitted exponents by kernel:', {k: f"{kernel_to_alpha[k]:.4f}" for k in kernel_list})
    return kernel_to_results, kernel_to_alpha


def run_cnn_padding_mode_experiment(dataset_name='cifar10', activation='relu', device='cuda'):
    """Compare circular vs zero padding under identical settings on CIFAR-10."""
    trainloader, _, input_channels, num_classes = get_data_loaders(dataset_name)
    depths = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30]
    lr_range = np.logspace(-3, 0, 50)

    mode_results = {}
    mode_alpha = {}
    for pad_mode in ['circular', 'zeros']:
        results = []
        for d in depths:
            kwargs = {
                'depth': d,
                'channels': 64,
                'kernel_size': 3,
                'num_classes': num_classes,
                'input_channels': input_channels,
                'activation': activation,
                'padding_mode': pad_mode,
            }
            best_lr, best_loss, _ = grid_search_lr(HomogeneousCNN, kwargs, trainloader, lr_range, device, num_trials=2, max_batches=None)
            results.append({'depth': d, 'best_lr': best_lr, 'best_loss': best_loss})
        mode_results[pad_mode] = results

        d_arr = np.array([r['depth'] for r in results])
        lr_arr = np.array([r['best_lr'] for r in results])
        reg = LinearRegression().fit(np.log(d_arr).reshape(-1, 1), np.log(lr_arr))
        mode_alpha[pad_mode] = reg.coef_[0]

    # Save results
    os.makedirs('results', exist_ok=True)
    for pad_mode in mode_results:
        pd.DataFrame(mode_results[pad_mode]).to_csv(f'results/cnn_padding_mode_results_{pad_mode}.csv', index=False)
    pd.DataFrame([
        {'mode': m, 'alpha': mode_alpha[m]} for m in mode_alpha
    ]).to_csv('results/cnn_padding_mode_summary.csv', index=False)

    # Plot
    plt.figure(figsize=(8, 6))
    colors = {'circular': 'tab:blue', 'zeros': 'tab:orange'}
    for pad_mode in ['circular', 'zeros']:
        results = mode_results[pad_mode]
        d_arr = np.array([r['depth'] for r in results])
        lr_arr = np.array([r['best_lr'] for r in results])
        plt.scatter(d_arr, lr_arr, s=60, alpha=0.7, color=colors[pad_mode], label=f'{pad_mode} data')
        reg = LinearRegression().fit(np.log(d_arr).reshape(-1, 1), np.log(lr_arr))
        k_fit = np.exp(reg.intercept_); a_fit = reg.coef_[0]
        d_grid = np.linspace(min(d_arr), max(d_arr), 100)
        plt.plot(d_grid, k_fit * (d_grid ** a_fit), '--', color=colors[pad_mode], linewidth=2, label=f'{pad_mode} fit: α={a_fit:.3f}')

    # Reference slope -1.5
    all_d = np.array([r['depth'] for r in mode_results['circular']])
    all_lr = np.array([r['best_lr'] for r in mode_results['circular']])
    k_theory = all_lr[0] * (all_d[0] ** 1.5)
    d_grid = np.linspace(min(all_d), max(all_d), 100)
    plt.plot(d_grid, k_theory * (d_grid ** (-1.5)), 'k-.', linewidth=2, label='Theory α=-1.5')

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Depth L'); plt.ylabel('Optimal LR η*')
    plt.title(f'CNN padding mode comparison on {dataset_name.upper()} ({activation.upper()})')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('cnn_padding_mode_compare.png', dpi=150); plt.show()

    print('Fitted exponents:', {m: f"{mode_alpha[m]:.4f}" for m in mode_alpha})
    return mode_results, mode_alpha


def run_resnet_depth_definition_compare(dataset_name='cifar10', activation='relu', device='cuda'):
    """Compare fitted lines when using our depth (n blocks) vs alternative depth
    defined as the sum of lengths of all 2^n paths through residual blocks.
    For n blocks, alternative depth is n * 2^{n-1}.
    """
    trainloader, _, input_channels, num_classes = get_data_loaders(dataset_name)
    depths = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30]
    lr_range = np.logspace(-3, 0, 50)

    results = []
    for n in depths:
        kwargs = {
            'depth': n,
            'num_classes': num_classes,
            'input_channels': input_channels,
            'activation': activation,
        }
        best_lr, best_loss, _ = grid_search_lr(PreActResNet, kwargs, trainloader, lr_range, device, num_trials=2, max_batches=None)
        results.append({'blocks': n, 'best_lr': best_lr})

    n_arr = np.array([r['blocks'] for r in results])
    lr_arr = np.array([r['best_lr'] for r in results])

    # Our depth definition: L_ours = n
    reg_ours = LinearRegression().fit(np.log(n_arr).reshape(-1, 1), np.log(lr_arr))
    k_ours = np.exp(reg_ours.intercept_); a_ours = reg_ours.coef_[0]

    # Alternative depth definition: L_alt = n * 2^{n-1}
    L_alt = n_arr * (2 ** (n_arr - 1))
    reg_alt = LinearRegression().fit(np.log(L_alt).reshape(-1, 1), np.log(lr_arr))
    k_alt = np.exp(reg_alt.intercept_); a_alt = reg_alt.coef_[0]

    # Save results
    os.makedirs('results', exist_ok=True)
    df_res = pd.DataFrame({'blocks': n_arr, 'best_lr': lr_arr, 'L_alt': L_alt})
    df_res.to_csv('results/resnet_depth_compare_results.csv', index=False)
    pd.DataFrame([
        {'method': 'ours', 'k': k_ours, 'alpha': a_ours},
        {'method': 'alt_sum_paths', 'k': k_alt, 'alpha': a_alt},
    ]).to_csv('results/resnet_depth_compare_summary.csv', index=False)

    # Plot comparison on one figure
    plt.figure(figsize=(8, 6))
    plt.scatter(n_arr, lr_arr, s=70, alpha=0.8, color='tab:blue', label='ResNet data (vs blocks)')
    d_grid = np.linspace(min(n_arr), max(n_arr), 200)
    plt.plot(d_grid, k_ours * (d_grid ** a_ours), '--', color='tab:blue', linewidth=2, label=f'Our depth fit: α={a_ours:.3f}')

    # For alternative, reparameterize x-axis using L_alt but show with same n grid by mapping
    L_alt_grid = d_grid * (2 ** (d_grid - 1))
    plt.plot(d_grid, k_alt * (L_alt_grid ** a_alt), '-.', color='tab:orange', linewidth=2, label=f'Alt depth fit (sum paths): α={a_alt:.3f}')

    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Blocks n (x-axis uses n; orange uses L_alt(n))'); plt.ylabel('Optimal LR η*')
    plt.title('ResNet depth definition comparison')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('resnet_depth_definition_compare.png', dpi=150); plt.show()

    return {
        'results': results,
        'ours': {'k': k_ours, 'alpha': a_ours},
        'alt': {'k': k_alt, 'alpha': a_alt},
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
    lr_range = np.logspace(-3, 0, 50) 
    
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
            max_batches=None
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
    if valid_results:
        plt.plot(all_depths_array, all_lrs_array, 'o-', markersize=8, linewidth=2)
    plt.xlabel('Depth L')
    plt.ylabel('Optimal Learning Rate η*')
    plt.title('Linear Scale View')
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


def run_imagenet_cnn_experiment(activation='relu', device='cuda'):
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
    lr_range = np.logspace(-3, 0, 50) 
    
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
            max_batches=None # Smaller batches for ImageNet
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
    if valid_results:
        plt.plot(all_depths_array, all_lrs_array, 'o-', markersize=8, linewidth=2)
    plt.xlabel('Depth L')
    plt.ylabel('Optimal Learning Rate η*')
    plt.title('Linear Scale View')
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


def run_imagenet_resnet_experiment(activation='relu', regularization='none', device='cuda'):
    """Run ResNet experiment on ImageNet with different regularization techniques"""
    
    print(f"\n{'='*60}")
    print(f"Running ImageNet ResNet experiment: {regularization.upper()} with {activation.upper()}")
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
    lr_range = np.logspace(-3, 0, 50) 
    
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
            max_batches=None  # Smaller batches for ImageNet
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
    plt.title(f'ImageNet ResNet {regularization.upper()} with {activation.upper()}')
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
    if valid_results:
        plt.plot(all_depths_array, all_lrs_array, 'o-', markersize=8, linewidth=2)
    plt.xlabel('Depth L')
    plt.ylabel('Optimal Learning Rate η*')
    plt.title('Linear Scale View')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'imagenet_resnet_{regularization}_{activation}.png', dpi=150)
    plt.show()
    
    return {
        'model_type': 'resnet',
        'activation': activation,
        'dataset': 'imagenet',
        'regularization': regularization,
        'all_results': all_results,
        'segment_results': segment_results,
        'global_alpha': global_alpha if valid_results else None,
        'theoretical_alpha': theoretical_alpha
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
    # # Choose which experiment to run:
    
    # # Option 1: Quick test (for debugging)
    # # results = quick_test()
    
    # # Option 2: Single detailed experiment with segmented baselines
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # # Test CNN with ReLU
    # # result = run_segmented_experiment('cnn', 'cifar10', 'relu', device)
    
    # # # Test CNN with GELU
    # # result = run_segmented_experiment('cnn', 'cifar10', 'gelu', device)
    
    # # Test ResNet with ReLU
    # # result = run_segmented_experiment('resnet', 'cifar10', 'relu', device)
    # # result = run_segmented_experiment('resnet', 'cifar100', 'relu', device)
    
    # # ==================== New Experiments ====================
    
    # # Experiment 1: 1D CNN on audio datasets
    # print("="*80)
    # print("EXPERIMENT 1: 1D CNN on Audio Datasets")
    # print("="*80)
    
    # # # Test 1D CNN on Google Speech Commands v2
    # # result = run_audio_experiment('1dcnn', 'speech_commands', 'relu', device)
    # # result = run_audio_experiment('1dcnn', 'speech_commands', 'gelu', device)
    
    # # # Test 1D CNN on ESC-50
    # # result = run_audio_experiment('1dcnn', 'esc50', 'relu', device)
    # # result = run_audio_experiment('1dcnn', 'esc50', 'gelu', device)
    
    # # Experiment 2: ResNet with regularization on CIFAR
    # print("="*80)
    # print("EXPERIMENT 2: ResNet with Regularization on CIFAR")
    # print("="*80)
    
    # # # Test ResNet with different regularization on CIFAR-10
    # # result = run_resnet_regularization_experiment('cifar10', 'relu', 'none', device)
    # # result = run_resnet_regularization_experiment('cifar10', 'relu', 'dropout', device)
    # # result = run_resnet_regularization_experiment('cifar10', 'relu', 'batchnorm', device)
    # # result = run_resnet_regularization_experiment('cifar10', 'relu', 'both', device)
    
    # # # Test ResNet with different regularization on CIFAR-100
    # # result = run_resnet_regularization_experiment('cifar100', 'relu', 'none', device)
    # # result = run_resnet_regularization_experiment('cifar100', 'relu', 'dropout', device)
    # # result = run_resnet_regularization_experiment('cifar100', 'relu', 'batchnorm', device)
    # # result = run_resnet_regularization_experiment('cifar100', 'relu', 'both', device)
    
    # # Experiment 3: ImageNet Experiments
    # print("="*80)
    # print("EXPERIMENT 3: ImageNet Experiments")
    # print("="*80)
    
    # # Test 2D CNN on ImageNet
    # print("\n--- 2D CNN on ImageNet ---")
    # # result = run_imagenet_cnn_experiment('relu', device)
    # # # result = run_imagenet_cnn_experiment('gelu', device)
    
    # # # Test ResNet on ImageNet with different regularization
    # # print("\n--- ResNet on ImageNet ---")
    # # result = run_imagenet_resnet_experiment('relu', 'none', device)
    # result = run_imagenet_resnet_experiment('relu', 'dropout', device)
    # # result = run_imagenet_resnet_experiment('relu', 'batchnorm', device)
    # result = run_imagenet_resnet_experiment('relu', 'both', device)
    
    # result = run_imagenet_resnet_experiment('gelu', 'none', device)
    # result = run_imagenet_resnet_experiment('gelu', 'dropout', device)
    # result = run_imagenet_resnet_experiment('gelu', 'batchnorm', device)
    # result = run_imagenet_resnet_experiment('gelu', 'both', device)
    
    # Option 3: Run all original experiments
    # all_results = run_all_experiments()

    # ==================== User Requested Experiments Calls ====================

    # print("\n" + "="*80)
    # print("EXPERIMENT 4: CNN Kernel Size Comparison on CIFAR-10")
    # print("="*80)
    # _ = run_cnn_kernel_size_experiment('cifar10', 'relu', device)

    print("\n" + "="*80)
    print("EXPERIMENT 5: CNN Padding Mode Comparison on CIFAR-100")
    print("="*80)
    _ = run_cnn_padding_mode_experiment('cifar100', 'relu', device)

    print("\n" + "="*80)
    print("EXPERIMENT 6: ResNet Depth Definition Comparison on CIFAR-10")
    print("="*80)
    _ = run_resnet_depth_definition_compare('cifar10', 'relu', device)
    print("\n" + "="*80)
    print("EXPERIMENT 6: ResNet Depth Definition Comparison on CIFAR-10")
    print("="*80)
    _ = run_resnet_depth_definition_compare('cifar100', 'relu', device)