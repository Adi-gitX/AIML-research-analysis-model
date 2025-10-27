"""Load a model checkpoint and predict tone + palette for a single image."""
import argparse
import json
from PIL import Image
import torch
from torchvision import transforms, models


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
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--img', required=True)
    parser.add_argument('--classes', required=True, help='path to classes.json or comma-separated list')
    parser.add_argument('--palette', default='/content/fashion_palette.json')
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    # load classes
    if args.classes.endswith('.json'):
        classes = json.load(open(args.classes))
    else:
        classes = args.classes.split(',')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, num_classes=len(classes), device=device)

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(args.img).convert('RGB')
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
    tone = classes[pred]
    palette = json.load(open(args.palette)) if args.palette and args.palette.endswith('.json') else {}
    print('Predicted tone:', tone)
    print('Recommended palette:', palette.get(tone, ['black', 'white']))


if __name__ == '__main__':
    main()
