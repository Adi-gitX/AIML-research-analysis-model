"""
Extract skin patches using the SkinToneClassifier (STONE).
Saves cropped patches to `out_dir/<label_name>/` for training.
"""
import os
import argparse
from pathlib import Path
from PIL import Image
import json

try:
    import stone
except Exception:
    stone = None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def process_image(img_path, out_dir, palette='monk'):
    # requires `skin-tone-classifier` installed and importable as `stone`
    res = stone.process(img_path, image_type='color', palette=palette, return_report_image=False)
    if not res or not res.get('faces'):
        return None
    face = res['faces'][0]
    # stone returns bounding box info in face['facebox'] (x,y,w,h) depending on version
    bbox = face.get('facebox')
    if not bbox:
        return None
    x, y, w, h = bbox
    img = Image.open(img_path).convert('RGB')
    # expand bbox slightly to include cheek area
    exp = int(0.2 * max(w, h))
    left = max(0, x - exp)
    top = max(0, y - exp)
    right = min(img.width, x + w + exp)
    bottom = min(img.height, y + h + exp)
    crop = img.crop((left, top, right, bottom))

    tone_label = face.get('tone_label') or face.get('tone') or face.get('skin_tone')
    # fallback label if not provided
    if not tone_label:
        tone_label = 'unknown'

    label_dir = Path(out_dir) / str(tone_label)
    ensure_dir(label_dir)
    out_file = label_dir / (Path(img_path).stem + '.jpg')
    crop.save(out_file)
    return {'file': str(out_file), 'label': tone_label, 'hex': face.get('skin_tone')}


def main(args):
    if stone is None:
        raise RuntimeError('The package `skin-tone-classifier` (stone) is required. Run `pip install skin-tone-classifier[all]`')
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    records = []
    for p in input_dir.glob('**/*'):
        if p.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        try:
            r = process_image(str(p), str(out_dir), palette=args.palette)
            if r:
                records.append(r)
        except Exception as e:
            print('Error processing', p, e)
    # save csv/json of pseudo labels
    with open(out_dir / 'pseudo_labels.json', 'w') as f:
        json.dump(records, f, indent=2)
    print('Done. Extracted', len(records), 'patches to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--palette', default='monk')
    args = parser.parse_args()
    main(args)
