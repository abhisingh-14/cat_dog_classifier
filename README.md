# Cat vs Dog Classifier

A minimal web app that identifies whether an uploaded photo shows a cat or a dog,
using a compact convolutional neural network trained **from scratch** (no pretrained
ImageNet weights) on ~20,000 images.

## Results

- Trained for 8 epochs on a single CPU core, ~64x64 input images.
- **Validation accuracy: 85.3%**
- **Held-out test accuracy: 84.2%** (5,000 images, never seen during training)

## What's included

```
cat_dog_classifier/
├── cat_dog_model_best.keras   # trained model weights
├── class_names.json           # ["cats", "dogs"] - class index mapping
├── train_state.json           # training history (loss/accuracy per epoch)
├── train_epoch.py             # the training script (for reference/retraining)
├── requirements.txt
└── webapp/
    ├── app.py                 # Flask server
    └── templates/index.html   # upload UI
```

## Running the app locally

1. Install dependencies (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:
   ```bash
   cd webapp
   python app.py
   ```
3. Open **http://127.0.0.1:5000** in your browser, upload a photo of a cat or dog,
   and click "Identify".

## How it was built

- **Dataset**: 25,000 labeled cat/dog images (20,000 for train/validation, 5,000 held
  out for testing), sourced from a public GitHub mirror of the classic Kaggle
  "Dogs vs. Cats" dataset.
- **Model**: a compact CNN (4 conv blocks with BatchNorm, ~550K parameters), trained
  from scratch since no internet access was available to download pretrained weights
  (e.g. MobileNetV2/ImageNet weights).
- **Training environment constraint**: single CPU core, so each epoch was run to
  completion, checkpointed, and resumed — `train_epoch.py` trains exactly one epoch
  per run and saves progress to `train_state.json` + a `.keras` checkpoint, so
  training can be safely stopped and resumed at any point.

## Retraining or improving the model

- Re-run `train_epoch.py` repeatedly (once per epoch) to continue training further —
  it will pick up from the last saved checkpoint automatically.
- To push accuracy higher, you could: increase image resolution, add more
  convolutional filters, train more epochs, or (if internet access is available)
  swap in transfer learning with a pretrained MobileNetV2 backbone.
