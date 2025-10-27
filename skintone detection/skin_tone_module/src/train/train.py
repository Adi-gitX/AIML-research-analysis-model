"""
Starter PyTorch training script for skin-tone classification.
Usage:
  python src/train/train.py --data_dir data/processed --epochs 10 --batch 32
"""
import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from src.utils.dataset import get_transforms, FolderDataset


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_tf = get_transforms(size=args.size, train=True)
    val_tf = get_transforms(size=args.size, train=False)

    train_ds = FolderDataset(os.path.join(args.data_dir), transform=train_tf)
    # For quick start we reuse same folder for val; you should create split
    val_ds = FolderDataset(os.path.join(args.data_dir), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    num_classes = len(train_ds.classes)
    print('Found classes:', train_ds.classes)

    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    # adjust final layer
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            total += imgs.size(0)
        epoch_loss = running / max(total, 1)
        print(f'Epoch {epoch+1}/{args.epochs} Train loss: {epoch_loss:.4f}')

        # quick val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)
        acc = correct / max(total, 1)
        print(f'Val acc: {acc:.4f}')
        # save best
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print('Saved best model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='models')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--size', type=int, default=128)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
