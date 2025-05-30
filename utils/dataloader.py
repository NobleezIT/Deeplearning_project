from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_dataloaders(data_dir, batch_size):
    """
    Loads training, validation, and test datasets using ImageFolder and returns corresponding DataLoaders.

    Args:
        data_dir (str): Path to the directory containing 'train', 'val', and 'test' subdirectories.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
            - train_loader (DataLoader): DataLoader for training set.
            - val_loader (DataLoader): DataLoader for validation set.
            - test_loader (DataLoader): DataLoader for test set.
            - class_names (list): Sorted list of class names.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train_dataset.classes

def get_class_names(data_dir):
    """
    Retrieves sorted list of class names based on subdirectory names in the training dataset.

    Args:
        data_dir (str): Path to the dataset directory containing a 'train' subdirectory.

    Returns:
        list: Sorted list of class names.
    """
    train_dir = os.path.join(data_dir, 'train')
    return sorted(entry.name for entry in os.scandir(train_dir) if entry.is_dir())
