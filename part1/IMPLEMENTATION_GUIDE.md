# Part 1: Detailed Implementation Guide

## 🎯 Overview

This guide provides step-by-step instructions for implementing the Oxford Pet Dataset classification system. You'll work with two pre-trained models (ResNet-50 and Swin-T) and two training strategies (feature extraction and full fine-tuning).

## 📚 Learning Objectives

By completing this implementation, you will understand:
- **Transfer Learning**: How to adapt pre-trained models for new tasks
- **Data Preprocessing**: Image augmentation and normalization techniques
- **Model Architecture**: Differences between CNN and Transformer architectures
- **Training Strategies**: Feature extraction vs. full fine-tuning
- **Evaluation**: Comprehensive model analysis and error patterns

## 🏗️ Implementation Structure

The project is organized into several modules, each with specific functions you need to implement:

1. **`data_utils.py`**: Data loading and preprocessing
2. **`model_utils.py`**: Model setup and configuration
3. **`training_utils.py`**: Training and evaluation loops
4. **`experiment_tracker.py`**: Experiment logging and visualization

---

## 📁 Module 1: Data Utilities (`data_utils.py`)


### Function 1: `load_datasets(data_dir='./data', image_size=224, batch_size=32, num_workers=4)`

**Purpose**: Load the Oxford Pet Dataset with proper train/validation splits.

**What you need to implement**:
```python
def load_datasets(data_dir='./data', image_size=224, batch_size=32, num_workers=4):
    """
    Load Oxford Pet Dataset with appropriate transforms.
    
    This function:
    1. Creates train and validation transforms
    2. Loads the Oxford Pet Dataset with proper splits
    3. Creates DataLoaders for efficient batch processing
    
    Dataset Information:
    - 37 different pet breeds
    - ~200 images per breed
    - Total: ~7,400 images
    - Split: trainval (training) and test (validation)
    
    Args:
        data_dir (str): Directory to store/load dataset
        image_size (int): Size to resize images to
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        tuple: (train_loader, val_loader, num_classes)
    """
    
```

**Key Learning Points**:
- **Why different splits?** trainval for training, test for validation
- **Why shuffle training but not validation?** Training benefits from randomness
- **Why pin_memory?** Faster data transfer to GPU

---

## 🏗️ Module 2: Model Utilities (`model_utils.py`)

### Function 1: `setup_resnet50(num_classes, pretrained=True)`

**Purpose**: Configure ResNet-50 for pet classification.

**What you need to implement**:
```python
def setup_resnet50(num_classes, pretrained=True):
    """
    Setup ResNet-50 model for fine-tuning.
    
    ResNet-50 Architecture:
    - 50-layer deep residual network
    - Pre-trained on ImageNet (1.2M images, 1000 classes)
    - Final layer: 1000 classes -> needs to be changed to 37 pet classes
    
    Key Steps:
    1. Load pre-trained ResNet-50
    2. Replace final classification layer
    3. Keep all other layers frozen initially
    
    Args:
        num_classes (int): Number of output classes (37 for pets)
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: ResNet-50 model ready for fine-tuning
    """
    
```

**Key Learning Points**:
- **Why replace final layer?** Pre-trained model was trained for 1000 ImageNet classes
- **Why keep other layers?** Lower layers learn general features (edges, textures)
- **What is fc?** "Fully Connected" - the final classification layer

### Function 2: `setup_swin_t(num_classes, pretrained=True)`

**Purpose**: Configure Swin Transformer for pet classification.

**What you need to implement**:
```python
def setup_swin_t(num_classes, pretrained=True):
    """
    Setup Swin Transformer (Swin-T) model for fine-tuning.
    
    Swin-T Architecture:
    - Transformer-based model (not CNN like ResNet)
    - Uses "shifted windows" for efficient attention computation
    - Pre-trained on ImageNet
    - Final layer: 1000 classes -> needs to be changed to 37 pet classes
    
    Key Differences from ResNet:
    - Uses attention mechanisms instead of convolutions
    - Better at capturing long-range dependencies
    - More parameters but often better performance
    
    Args:
        num_classes (int): Number of output classes (37 for pets)
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: Swin-T model ready for fine-tuning
    """
    
```

**Key Learning Points**:
- **Why different layer name?** Swin-T uses 'head' instead of 'fc'
- **What are transformers?** Attention-based models that process sequences
- **Why Swin-T?** More efficient than full transformers, better than CNNs

### Function 3: `freeze_backbone(model, model_name)`

**Purpose**: Freeze all layers except the final classification layer for feature extraction.

**What you need to implement**:
```python
def freeze_backbone(model, model_name):
    """
    Freeze all parameters except the final classification layer.
    
    This implements "Feature Extraction" strategy:
    - Keep pre-trained features frozen
    - Only train the final classification layer
    - Faster training, less prone to overfitting
    - Good for small datasets
    
    Freezing Process:
    1. Set requires_grad=False for all parameters
    2. Set requires_grad=True only for final layer
    
    Args:
        model (torch.nn.Module): The model to freeze
        model_name (str): Name of the model ('resnet50' or 'swin_t')
    
    Returns:
        torch.nn.Module: Model with frozen backbone
    """
    
```

**Key Learning Points**:
- **Why freeze backbone?** Pre-trained features are already good
- **What is requires_grad?** Controls whether parameters are updated during training
- **Why different layer names?** Different architectures use different naming

---

## 🏋️ Module 3: Training Utilities (`training_utils.py`)

### Function 1: `train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None)`

**Purpose**: Train the model for one complete epoch.

**What you need to implement**:
```python
def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    """
    Train the model for one epoch.
    
    Training Loop Steps:
    1. Set model to training mode
    2. For each batch:
       a. Move data to device (GPU/CPU)
       b. Zero gradients
       c. Forward pass (compute predictions)
       d. Compute loss
       e. Backward pass (compute gradients)
       f. Update parameters
    3. Calculate average loss and accuracy
    4. Update learning rate scheduler
    
    Args:
        model (torch.nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to run on
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    
```

**Key Learning Points**:
- **Why model.train()?** Enables training-specific behaviors (dropout, batch norm)
- **Why zero_grad()?** PyTorch accumulates gradients, must clear them
- **Why .to(device)?** Data must be on same device as model (GPU/CPU)

### Function 2: `validate_epoch(model, val_loader, criterion, device)`

**Purpose**: Validate the model for one epoch.

**What you need to implement**:
```python
def validate_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch.
    
    Validation Loop Steps:
    1. Set model to evaluation mode
    2. Disable gradient computation (torch.no_grad())
    3. For each batch:
       a. Move data to device
       b. Forward pass (no backward pass needed)
       c. Compute loss and predictions
       d. Track statistics
    4. Calculate average metrics
    
    Key Differences from Training:
    - No gradient computation (faster)
    - No parameter updates
    - Model in evaluation mode
    
    Args:
        model (torch.nn.Module): The model to validate
        val_loader (DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to run on
    
    Returns:
        tuple: (average_loss, accuracy, all_predictions, all_targets)
    """
    
```

**Key Learning Points**:
- **Why model.eval()?** Disables training-specific behaviors
- **Why torch.no_grad()?** Saves memory, no gradients needed for validation
- **Why store predictions?** Needed for detailed analysis (confusion matrix, etc.)

---

## 🚀 Running the Complete Experiment

### Step 1: Test Individual Components

```python
# Test data loading
from data_utils import load_datasets, get_transforms
train_loader, val_loader, num_classes = load_datasets()
print(f"Loaded {num_classes} classes")

# Test model setup
from model_utils import setup_resnet50, setup_swin_t
model = setup_resnet50(num_classes)
print(f"ResNet-50 created with {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Step 2: Run Training

```python
# Run the main experiment
python main.py --num_epochs 5 --batch_size 32 --use_wandb
```

### Step 3: Analyze Results

The experiment will generate:
- Training history plots
- Confusion matrices
- Comparison plots
- Results table
- Error analysis

---

## 🎯 Expected Results

After successful implementation, you should see:

1. **Training Progress**: Loss decreasing, accuracy increasing
2. **Model Comparison**: Different performance between ResNet-50 and Swin-T
3. **Strategy Comparison**: Feature extraction vs. full fine-tuning
4. **Error Patterns**: Which pet breeds are most challenging

## 🐛 Common Issues and Solutions

### Issue 1: CUDA Out of Memory
**Solution**: Reduce batch size
```python
python main.py --batch_size 16
```

### Issue 2: Slow Training
**Solution**: Use GPU and increase num_workers
```python
python main.py --device cuda --num_workers 8
```

### Issue 3: Poor Performance
**Solution**: Check data preprocessing and learning rates
- Ensure ImageNet normalization is correct
- Try different learning rates (1e-2 for feature extraction, 1e-3 for fine-tuning)

## 📚 Key Concepts Summary

1. **Transfer Learning**: Using pre-trained models for new tasks
2. **Feature Extraction**: Freezing backbone, training only final layer
3. **Fine-tuning**: Training entire network with lower learning rates
4. **Data Augmentation**: Random transformations to increase dataset diversity
5. **Model Evaluation**: Comprehensive analysis of performance and errors

This implementation will give you hands-on experience with modern deep learning techniques and prepare you for advanced computer vision projects!
