"""
Main script to train and evaluate a deep learning model (ResNet or EfficientNet) 
for classifying Nigerian agricultural produce. Supports configurable training parameters.
"""

import argparse
import os
from utils.train import train_model
from utils.evaluate import evaluate_model
from models.resnet_model import get_resnet
from models.efficientnet_model import get_efficientnet
from utils.dataloader import get_dataloaders

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate a model on Nigerian produce dataset.")
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'efficientnet'],
                        help='Model type to use for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--data_dir', type=str, default='data/split_dataset',
                        help='Directory containing the split dataset')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    args = parser.parse_args()

    # Load data loaders and class names
    train_loader, val_loader, test_loader, class_names = get_dataloaders(args.data_dir, args.batch_size)

    # Initialize model
    if args.model == 'resnet':
        model = get_resnet(len(class_names))
    else:
        model = get_efficientnet(len(class_names))

    # Train model
    best_model_path = train_model(model, train_loader, val_loader, args)

    # Evaluate model if training was successful
    if best_model_path and os.path.exists(best_model_path):
        evaluate_model(best_model_path, test_loader, class_names)
    else:
        print("No model was saved during training. Evaluation skipped.")
