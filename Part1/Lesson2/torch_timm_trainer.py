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
from typing import Tuple, Callable
from dataclasses import dataclass
from tqdm.auto import tqdm
import time

@dataclass
class TrainingConfig:
    model_name: str = 'resnet18'
    batch_size: int = 32
    num_epochs: int = 5
    learning_rate: float = 0.0001 # 0.00005
    image_size: Tuple[int, int] = (224, 224)
    num_workers: int = 4

class DatasetSplitter:
    def __init__(self, root_path: Path, destination_path: Path, valid_fraction: float = 0.2):
        """Split a dataset into training and validation sets.
        Args:
            root_path (Path): Path to the root of the dataset.
            destination_path (Path): Path to the destination folder.
            valid_fraction (float, optional): Fraction of the data to use for validation. Defaults to 0.2.
        """
        self.root_path = root_path
        self.destination_path = destination_path
        self.valid_fraction = valid_fraction
        self.train_path = self.destination_path / '_train'
        self.valid_path = self.destination_path / '_valid'

    def is_dataset_split(self) -> bool:
        """Check if the dataset is already correctly split by validating:
        1. Existence of train/valid directories
        2. Matching class names across directories
        3. Correct validation fraction for each class
        """
        if not (self.train_path.exists() and self.valid_path.exists()):
            return False

        # Verify class integrity
        original_classes = {p.name for p in self.root_path.iterdir() if p.is_dir()}
        train_classes = {p.name for p in self.train_path.iterdir() if p.is_dir()}
        valid_classes = {p.name for p in self.valid_path.iterdir() if p.is_dir()}
        
        if not (original_classes == train_classes == valid_classes):
            return False

        # Verify validation fraction for each class
        for class_name in original_classes:
            train_count = len(list((self.train_path / class_name).glob('*.*')))
            valid_count = len(list((self.valid_path / class_name).glob('*.*')))
            total_count = train_count + valid_count
            
            actual_fraction = valid_count / total_count if total_count > 0 else 0
            # Allow for small rounding differences (±1%)
            if abs(actual_fraction - self.valid_fraction) > 0.01:
                return False

        return True


    def split(self) -> Tuple[Path, Path]:
        """Split the dataset into training and validation sets.
        Returns:
            Tuple[Path, Path]: Paths to the training and validation directories.
        """
        if self.is_dataset_split():
            print("Dataset already correctly split")
            return self.train_path, self.valid_path

        print("Splitting dataset...")
        self.train_path.mkdir(exist_ok=True, parents=True)
        self.valid_path.mkdir(exist_ok=True, parents=True)

        for class_folder in self.root_path.iterdir():
            if class_folder.is_dir():
                (self.train_path / class_folder.name).mkdir(exist_ok=True)
                (self.valid_path / class_folder.name).mkdir(exist_ok=True)
                
                image_files = list(class_folder.glob('*.*'))
                train_files, valid_files = train_test_split(
                    image_files, 
                    test_size=self.valid_fraction,
                    random_state=42
                )
                
                for file in train_files:
                    shutil.copy2(file, self.train_path / class_folder.name / file.name)
                for file in valid_files:
                    shutil.copy2(file, self.valid_path / class_folder.name / file.name)

        return self.train_path, self.valid_path

class TrainingCallback:
    def on_training_start(self, trainer: 'ModelTrainer'): pass
    def on_epoch_start(self, trainer: 'ModelTrainer', epoch: int): pass
    def on_batch_end(self, trainer: 'ModelTrainer', epoch: int, batch: int, loss: float): pass
    def on_epoch_end(self, trainer: 'ModelTrainer', epoch: int, avg_loss: float): pass
    def on_training_end(self, trainer: 'ModelTrainer'): pass

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.callbacks = []
        self.best_loss = float('inf')
        
    def setup_data(self, train_path: Path, valid_path: Path):
        transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_dataset = ImageFolder(root=train_path, transform=transform)
        self.valid_dataset = ImageFolder(root=valid_path, transform=transform)
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers
        )
        self.valid_loader = DataLoader(
            self.valid_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers
        )

    def setup_model(self):
        self.model = timm.create_model(self.config.model_name, pretrained=True, 
                                     num_classes=len(self.train_dataset.classes))
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

    def add_callback(self, callback: TrainingCallback):
        self.callbacks.append(callback)

    def train(self):
        for callback in self.callbacks:
            callback.on_training_start(self)

        for epoch in range(self.config.num_epochs):
            for callback in self.callbacks:
                callback.on_epoch_start(self, epoch)

            self.model.train()
            total_loss = 0
            steps = 0

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                steps += 1

                for callback in self.callbacks:
                    callback.on_batch_end(self, epoch, batch_idx, loss.item())

            avg_loss = total_loss / steps
            
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, avg_loss)

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(epoch, avg_loss)

        for callback in self.callbacks:
            callback.on_training_end(self)

    def save_checkpoint(self, epoch: int, loss: float):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, f"model_torch_timm_epoch_{epoch+1}.pth")

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

# class PrintCallback(TrainingCallback):
#     def on_batch_end(self, trainer: ModelTrainer, epoch: int, batch: int, loss: float):
#         if (batch + 1) % 10 == 0:
#             print(f"Epoch [{epoch+1}/{trainer.config.num_epochs}], "
#                   f"Step [{batch+1}/{len(trainer.train_loader)}], Loss: {loss:.4f}")
    
#     def on_epoch_end(self, trainer: ModelTrainer, epoch: int, avg_loss: float):
#         print(f"Epoch [{epoch+1}/{trainer.config.num_epochs}] completed, "
#               f"Average Loss: {avg_loss:.4f}")

class ProgressCallback(TrainingCallback):
    def on_training_start(self, trainer: ModelTrainer):
        print("\nTraining Started")
        print("-" * 80)
        
    def on_epoch_start(self, trainer: ModelTrainer, epoch: int):
        self.batch_bar = tqdm(total=len(trainer.train_loader), 
                            desc=f'Epoch {epoch+1}/{trainer.config.num_epochs}', 
                            leave=False, unit='batch')
        self.epoch_start_time = time.time()
        self.running_loss = 0
        
    def on_batch_end(self, trainer: ModelTrainer, epoch: int, batch: int, loss: float):
        self.running_loss = 0.9 * self.running_loss + 0.1 * loss
        self.batch_bar.update()
        
    def on_epoch_end(self, trainer: ModelTrainer, epoch: int, avg_loss: float):
        accuracy = trainer.validate()
        self.batch_bar.close()
        
        epoch_time = time.time() - self.epoch_start_time
        iterations_per_sec = len(trainer.train_loader) / epoch_time
        
        epoch_width = len(str(trainer.config.num_epochs))
        print(f"Epoch {epoch+1:>{epoch_width}d}/{trainer.config.num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val Acc: {accuracy:>6.2f}% | "
              f"Time: {epoch_time:>5.1f}s | "
              f"It/s: {iterations_per_sec:>4.1f} | "
              f"Best Loss: {trainer.best_loss:.4f}")

def main():
    # Setup paths and configuration
    dataset_root = Path(r"Part1\Lesson1\bird_dataset")
    dataset_dest = Path(r"datasets\bird_dataset_split")
    config = TrainingConfig()

    # Split dataset
    splitter = DatasetSplitter(dataset_root, dataset_dest)
    train_path, valid_path = splitter.split()

    # Setup trainer
    trainer = ModelTrainer(config)
    trainer.setup_data(train_path, valid_path)
    trainer.setup_model()
    trainer.add_callback(ProgressCallback())

    # Train model
    trainer.train()

    # Validate
    accuracy = trainer.validate()
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # Release device memory
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

if __name__ == '__main__':
    main()
