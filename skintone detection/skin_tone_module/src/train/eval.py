"""Evaluation utilities: compute confusion matrix and per-class metrics."""
import argparse
import os
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def load_model(ckpt, num_classes, device):
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = datasets.ImageFolder(args.data_dir, transform=tf)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    model = load_model(args.ckpt, num_classes=len(dataset.classes), device=device)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    print('\nClassification report:')
    print(classification_report(y_true, y_pred, target_names=dataset.classes, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print('\nConfusion matrix:\n', cm)


if __name__ == '__main__':
    main()
