"""
This module defines a training loop for image classification using PyTorch.

It supports TensorBoard logging, early stopping, and saving the best model based on validation accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

def train_model(model, train_loader, val_loader, args):
    """
    Trains a given model using provided training and validation data loaders.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        args (argparse.Namespace): Training arguments including:
            - epochs (int): Number of training epochs.
            - lr (float): Learning rate.
            - model (str): Model type name (used for naming log files).
            - patience (int): Number of epochs to wait before early stopping if no improvement.

    Returns:
        str or None: Path to the best saved model file, or None if no model was saved.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log_dir = f'logs/{args.model}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(log_dir=log_dir)

    best_acc = 0.0
    best_model_path = f'saved_models/best_{args.model}.pth'
    os.makedirs('saved_models', exist_ok=True)

    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch [{epoch+1}/{args.epochs}], Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    writer.close()

    if os.path.exists(best_model_path):
        return best_model_path
    else:
        print("No best model was saved. Validation may not have improved.")
        return None
