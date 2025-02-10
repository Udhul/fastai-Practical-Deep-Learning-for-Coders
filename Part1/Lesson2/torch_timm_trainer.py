#---Imports
import torch
import timm
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

#---Constants
DATASET_ROOT = Path(r"Part1\Lesson1\bird_dataset")
DATASET_DEST = Path(r"datasets\bird_dataset_split")

#---Dataset splitter
def split_dataset(root_path: Path, destination_path: Path = None, valid_fraction: float = 0.2):
    """
    Splits a dataset into training and validation sets based on labeled subfolders.
    
    Args:
        root_path: Path to the root directory containing class subfolders
        destination_path: Path to the destination directory (default is root_path.parent)
        valid_fraction: Fraction of data to use for validation (default 0.2 = 20%)
    
    Returns:
        train_path: Path to the training dataset
        valid_path: Path to the validation dataset
    """
    # Create train and valid directories
    if not destination_path:
        train_path = root_path / '_train'
        valid_path = root_path / '_valid'
    else:
        train_path = destination_path / '_train'
        valid_path = destination_path / '_valid'
    
    # Create directories if they don't exist
    train_path.mkdir(exist_ok=True)
    valid_path.mkdir(exist_ok=True)
    
    # Process each class folder
    for class_folder in root_path.iterdir():
        if class_folder.is_dir():
            # Create corresponding class folders in train and valid
            (train_path / class_folder.name).mkdir(exist_ok=True)
            (valid_path / class_folder.name).mkdir(exist_ok=True)
            
            # Get all image files
            image_files = list(class_folder.glob('*.*'))
            
            # Split into train and validation
            train_files, valid_files = train_test_split(
                image_files, 
                test_size=valid_fraction,
                random_state=42
            )
            
            # Copy files to respective directories
            for file in train_files:
                shutil.copy2(file, train_path / class_folder.name / file.name)
            for file in valid_files:
                shutil.copy2(file, valid_path / class_folder.name / file.name)
    
    return train_path, valid_path



if __name__ == '__main__':

    #---Data preparation
    train_dataset_path = DATASET_DEST / '_train'
    valid_dataset_path = DATASET_DEST / '_valid'

    print("train path:",train_dataset_path, "validation path:", valid_dataset_path)
    # train_dataset_path, valid_dataset_path = split_dataset(DATASET_ROOT, DATASET_DEST)

    # Define data augmentation and normalization (similar to FastAI's transforms)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset from folder (FastAI's `DataBlock` uses `ImageFolder` under the hood)
    train_dataset = ImageFolder(root=train_dataset_path, transform=transform)
    valid_dataset = ImageFolder(root=valid_dataset_path, transform=transform)

    # Define dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    #---Model definition
    # Load a pretrained ResNet model from timm
    model = timm.create_model('resnet34', pretrained=True, num_classes=len(train_dataset.classes))

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    #---Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Standard loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    #----Training loop
    # Training loop with progress tracking
    num_epochs = 5
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        steps = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights
            
            total_loss += loss.item()
            steps += 1
            
            # Print batch progress every 10 steps
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / steps
        print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}")
        
        # Save model if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f"model_torch_timm_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1} with loss: {best_loss:.4f}")


    #---Load model from saved file
    # model_loaded = torch.load("model_torch_timm.pth")

    #---Validation loop
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation during validation
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")
