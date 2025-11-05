# Facial-Expression-Recognition-under-occlusion
Overview

A Vision Transformer (ViT)-based model for facial expression recognition, robust to partial occlusions such as masks, glasses, or hands. Trained on the FER2013 dataset, the system combines advanced data augmentation and transfer learning to improve accuracy under real-world conditions.

⸻

Features
	•	Architecture: ViT-Base (timm)
	•	Dataset: FER2013 (7 emotion classes)
	•	Robustness: Occlusion simulation via random blocks and CoarseDropout
	•	Training: AdamW optimizer, cosine scheduler, mixed precision
	•	Evaluation: Accuracy, confusion matrix, optional attention visualization

⸻

Installation

git clone https://github.com/yourusername/facial-expression-vit.git
cd facial-expression-vit
pip install torch torchvision timm albumentations pandas scikit-learn matplotlib


⸻

Dataset

Download FER2013 CSV and place it in the project root.

⸻

Training

python train_vit_occlusion.py

	•	Defaults: epochs=25, batch_size=32, lr=3e-5, img_size=224
	•	Best model saved as best_vit_occlusion.pth

⸻

Evaluation

val_loss, val_acc = validate(model, val_loader, criterion, device)
print("Validation Accuracy:", val_acc)


⸻

Results

Model	Accuracy	Notes
ViT-Base (no occlusion)	~71%	Baseline
ViT-Base (occlusion-trained)	~77%	Robust to partial occlusion


⸻

Extensions
	•	Multi-dataset training (RAF-DB, AffectNet)
	•	Hybrid CNN-ViT fusion
	•	Explainability with attention heatmaps
	•	Deployment via Gradio/FastAPI

⸻

License

MIT License — free to use and modify with attribution.
