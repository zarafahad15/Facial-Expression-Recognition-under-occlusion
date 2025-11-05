# Facial-Expression-Recognition-under-occlusion
Overview

A Vision Transformer (ViT)-based model trained for facial expression recognition under partial occlusions (masks, glasses, hands). It enhances robustness using occlusion-aware augmentation and fine-tuning on the FER2013 dataset.

⸻

Key Details
	•	Model: ViT-Base (from timm)
	•	Dataset: FER2013 (48×48 grayscale faces)
	•	Tech: PyTorch, Albumentations, AdamW optimizer
	•	Features: Random occlusion, Cutout, Mixed precision training
	•	Output: Seven emotions — Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

⸻

Installation

git clone https://github.com/yourusername/facial-expression-vit.git
cd facial-expression-vit
pip install torch torchvision timm albumentations pandas scikit-learn matplotlib


⸻

Dataset

Download FER2013 and place fer2013.csv in the project root.

⸻

Training

python train_vit_occlusion.py

Default settings:
epochs=25, batch_size=32, lr=3e-5, img_size=224

The best model is saved as best_vit_occlusion.pth.

⸻

Evaluation

val_loss, val_acc = validate(model, val_loader, criterion, device)
print("Validation Accuracy:", val_acc)


⸻

Results

Model	Accuracy	Note
ViT-Base (no occlusion)	~71%	Baseline
ViT-Base (occlusion-trained)	~77%	Robust version


⸻

Future Work
	•	Multi-dataset training (RAF-DB, AffectNet)
	•	Hybrid CNN-ViT fusion
	•	Explainability via attention maps
	•	Gradio/FastAPI deployment

⸻

License

MIT License — free to use and modify with attribution.

⸻
