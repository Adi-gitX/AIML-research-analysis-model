# Skin Tone Module (for Fashion Recommendation)

This folder is a starter scaffold to build a trainable skin-tone classification module for your fashion recommendation pipeline.

Structure

- data/raw: place raw images here (user photos / dataset images)
- data/processed: extracted skin patches and labeled folders (for ImageFolder)
- src/stone_integration: scripts that use the SkinToneClassifier (STONE) for extraction/pseudo-labeling
- src/train: training scripts (PyTorch)
- src/utils: dataset and helper utilities
- notebooks: Colab / Jupyter notebook template for the full pipeline
- models: saved checkpoints and logs
- configs: training and dataset config files

Quick start

1. Install dependencies: see `requirements.txt`.
2. Put example images in `data/raw/`.
3. Run the extractor to create skin patches and pseudo-labels:
   python src/stone_integration/extract_patches.py --input_dir data/raw --out_dir data/processed
4. Train using the starter training script:
   python src/train/train.py --data_dir data/processed --epochs 10

Notes

- This scaffold uses the `skin-tone-classifier` package (STONE) only for preprocessing/pseudo-labeling. You will train your own CNN model on the extracted patches.
- Customize labels (Monk 1-10) by re-labeling `data/processed` folders or by using a mapping file.
