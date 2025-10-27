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

Production-ready commands

1. Create splits from pseudo-labels (optional):

   python src/utils/make_split.py --pseudo /path/to/pseudo_labels.json --out /path/to/data/dataset --seed 42 --val_ratio 0.15 --test_ratio 0.10

2. Train a production model (EfficientNet-B0):

   python src/train/train_full.py --data_dir /path/to/data/dataset --output_dir models/effnet_b0 --epochs 30 --batch 32 --img_size 224

3. Evaluate saved model:

   python src/train/eval.py --data_dir /path/to/data/dataset --ckpt models/effnet_b0/best_model.pt

4. Run inference on a single image:

   python src/inference/predict.py --ckpt models/effnet_b0/best_model.pt --img /path/to/img.jpg

5. Serve model (FastAPI):

   uvicorn src.serve.fastapi_app:app --host 0.0.0.0 --port 8000

If you plan to run in Colab use the notebook under `notebooks/skin_tone_training.ipynb` as a quick starter.
