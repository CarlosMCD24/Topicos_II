"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data with enhanced transformations for training.
"""
# Importa librerías necesarias
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    image_size: int,
    batch_size: int, 
    num_workers: int = NUM_WORKERS
):
    """
    Creates training and testing DataLoaders with data augmentation.
    """

    # -------------------------------
    # 🔥 Transformaciones mejoradas
    # -------------------------------

    # Transformaciones para entrenamiento (AUMENTO DE DATOS)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),       # ← rotación agregada
        transforms.ColorJitter(brightness=0.2, 
                               contrast=0.2, 
                               saturation=0.2, 
                               hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # Transformaciones para validación / test (SIN aumento de datos)
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # -------------------------------
    # 🔥 Datasets
    # -------------------------------
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Obtiene las clases
    class_names = train_data.classes

    # -------------------------------
    # 🔥 DataLoaders
    # -------------------------------
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names
