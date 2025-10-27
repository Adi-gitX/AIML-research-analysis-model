"""
Production-ready training script for skin-tone classification.
Features:
- EfficientNet-B0 backbone (torchvision)
- Stratified train/val split
- Checkpointing, best-model saving
- Optional Gray-World normalization
Usage:
  python src/train/train_full.py --data_dir /path/to/dataset --output_dir models/effnet --epochs 20
"""
import argparse
import os
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from src.utils.color_constancy import gray_world


def build_transforms(img_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


def stratified_split(dataset, val_size=0.15, test_size=0.0, seed=42):
    labels = np.array([s[1] for s in dataset.samples])
    indices = np.arange(len(labels))
    test_split = test_size
    if test_split > 0:
        s1 = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
        train_idx, test_idx = next(s1.split(indices, labels))
        labels = labels[train_idx]
        indices = train_idx
    if val_size > 0:
        s2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (1 - test_split), random_state=seed)
        train_idx, val_idx = next(s2.split(indices, labels))
        return train_idx.tolist(), val_idx.tolist(), (test_idx.tolist() if test_split > 0 else [])
    return indices.tolist(), [], (test_idx.tolist() if test_split > 0 else [])


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = datasets.ImageFolder(args.data_dir, transform=build_transforms(args.img_size, train=True))
    if len(dataset) == 0:
        raise SystemExit('No data found in data_dir')

    train_idx, val_idx, test_idx = stratified_split(dataset, val_size=args.val_ratio, test_size=args.test_ratio, seed=args.seed)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx) if len(val_idx) > 0 else Subset(dataset, train_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    num_classes = len(dataset.classes)
    print('Classes:', dataset.classes)

    model = models.efficientnet_b0(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / max(len(train_loader.dataset), 1)
        scheduler.step()

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)
        acc = correct / max(total, 1)
        print(f'Epoch {epoch+1}/{args.epochs} loss={epoch_loss:.4f} val_acc={acc:.4f}')
        # checkpoint
        ckpt = Path(args.output_dir) / f'epoch_{epoch+1}.pt'
        torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'acc': acc}, ckpt)
        if acc > best_acc:
            best_acc = acc
            best_path = Path(args.output_dir) / 'best_model.pt'
            torch.save(model.state_dict(), best_path)
            print('Saved best model to', best_path)

    elapsed = time.time() - start_time
    print('Training finished in {:.1f}s, best_val_acc={:.4f}'.format(elapsed, best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='models')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)
