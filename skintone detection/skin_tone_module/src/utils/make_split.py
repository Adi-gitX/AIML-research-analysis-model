"""Create train/val/test splits from a pseudo-label JSON created by the extractor.
Usage:
  python src/utils/make_split.py --pseudo /path/to/pseudo_labels.json --out /path/to/data/dataset --val_ratio 0.15 --test_ratio 0.10
"""
import argparse
import json
import os
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split


def load_pseudo(pseudo_json):
    with open(pseudo_json, 'r') as f:
        return json.load(f)


def build_class_map(records):
    classes = defaultdict(list)
    for r in records:
        tone = r.get('tone', 'tone_5')
        classes[tone].append(r['img'])
    return classes


def copy_split(out_root, splits):
    for split_name, mapping in splits.items():
        for cls, files in mapping.items():
            dest = os.path.join(out_root, split_name, cls)
            os.makedirs(dest, exist_ok=True)
            for src in files:
                try:
                    shutil.copy(src, dest)
                except Exception:
                    # ignore missing files
                    pass


def make_splits(pseudo_json, out_root, val_ratio=0.15, test_ratio=0.1, seed=42):
    recs = load_pseudo(pseudo_json)
    class_map = build_class_map(recs)
    train_map = {}
    val_map = {}
    test_map = {}
    for cls, files in class_map.items():
        if len(files) == 0:
            continue
        # if only one file, put into train
        if len(files) < 3:
            train_map[cls] = files
            val_map[cls] = []
            test_map[cls] = []
            continue
        train_files, temp = train_test_split(files, test_size=(val_ratio + test_ratio), random_state=seed)
        if val_ratio == 0 and test_ratio == 0:
            val_files, test_files = [], []
        else:
            # split temp into val/test proportionally
            rel_val = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0
            val_files, test_files = train_test_split(temp, test_size=(1 - rel_val), random_state=seed)
        train_map[cls] = train_files
        val_map[cls] = val_files
        test_map[cls] = test_files

    splits = {'train': train_map, 'val': val_map, 'test': test_map}
    copy_split(out_root, splits)
    print('Created splits under', out_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pseudo', required=True, help='pseudo_labels.json path')
    parser.add_argument('--out', required=True, help='output root for dataset splits')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    make_splits(args.pseudo, args.out, args.val_ratio, args.test_ratio, args.seed)


if __name__ == '__main__':
    main()
