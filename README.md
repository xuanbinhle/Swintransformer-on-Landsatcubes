
# Swin Transformer for Landsat Time-Series Classification

This repository trains a modified Swin Transformer with 6-channel Landsat cubes and performs multi-label classification on ecological survey data.

## 📁 Folder Structure

```
swin_landsat_module/
├── dataloaders/
│   └── dataset.py
├── models/
│   └── swin.py
├── trainer/
│   └── train.py
├── inference/
│   └── test.py
├── utils/
│   └── helpers.py
└── main.py
```

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install torch torchvision timm pandas scikit-learn
```

### 2. Run Training + Inference

```bash
python main.py \
    --train_data /path/to/train/cubes \
    --train_meta /path/to/train/meta.csv \
    --test_data /path/to/test/cubes \
    --test_meta /path/to/test/meta.csv \
    --epochs 20 \
    --lr 0.0002 \
    --batch_size 64 \
    --save_model_path swin-with-landsat-cubes.pth \
    --submission_path submission_swintransformer.csv
```

## ✅ Outputs

- `swin-with-landsat-cubes.pth` – trained model
- `submission_swintransformer.csv` – top-25 prediction per surveyId

## 🔧 Configuration

You can modify hyperparameters like:
- `--epochs`: training epochs
- `--batch_size`: training batch size
- `--lr`: learning rate
- `--num_classes`: default `11255`

## 🧠 Model

Modified Swin Transformer with 6-channel input adapted to Landsat bands (e.g., RGB, NIR, SWIR1, SWIR2...).

---

© 2025 EarthAI - GeoLifeCLEF Challenge
