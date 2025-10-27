from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os


def get_transforms(size=128, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])


class FolderDataset(Dataset):
    """Simple ImageFolder-like loader for preprocessed patches.
    Expects `root/<label_name>/*.jpg` structure.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.classes = []
        for d in sorted(os.listdir(root)):
            full = os.path.join(root, d)
            if os.path.isdir(full):
                self.classes.append(d)
                for f in os.listdir(full):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(full, f), d))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label]
