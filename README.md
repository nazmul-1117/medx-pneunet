# 📘 MedX-PneuNet: Explainable Deep Learning for Pneumonia Detection using Chest X-ray Images

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-red)

---

## 📌 Abstract

Pneumonia remains one of the leading causes of mortality worldwide, particularly in low-resource healthcare settings. Early and accurate diagnosis using chest X-ray imaging is critical for timely treatment. In this project, we propose **MedX-PneuNet**, an explainable deep learning framework for automated pneumonia detection using chest X-ray images.

The system leverages **transfer learning-based convolutional neural networks (CNNs)**, specifically **MobileNetV2 (baseline)** and **ResNet50 (comparative model)**, to perform binary classification between *Normal* and *Pneumonia* cases. To enhance clinical interpretability, we integrate **Gradient-weighted Class Activation Mapping (Grad-CAM)** for visual explainability.

Additionally, we conduct a **dataset efficiency study** and **model benchmarking analysis** to evaluate performance under full and reduced dataset conditions, simulating real-world low-resource scenarios.

---

## 🎯 Key Contributions

This work extends existing literature with the following contributions:

- ✔ Reproduction and validation of MobileNetV2-based pneumonia classification
- ✔ Comparative evaluation with ResNet50 architecture
- ✔ Dataset efficiency analysis (full vs reduced dataset)
- ✔ Comprehensive evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- ✔ Explainable AI integration using Grad-CAM for clinical interpretability
- ✔ Detailed error analysis of false positive and false negative predictions
- ✔ Lightweight and reproducible training pipeline suitable for GPU environments

---

## 🧠 Methodology Overview

The proposed pipeline consists of the following stages:

```

Chest X-ray Images
↓
Data Cleaning & Preprocessing
↓
Dataset Splitting (Train / Validation / Test)
↓
Data Augmentation
↓
Transfer Learning Models (MobileNetV2 / ResNet50)
↓
Fine-tuning (Optional)
↓
Model Training (GPU-based)
↓
Evaluation Metrics
↓
Grad-CAM Explainability
↓
Error Analysis
↓
Performance Comparison

````

---

## 🏗️ Model Architectures

### 🔹 MobileNetV2 (Baseline)
- Lightweight CNN architecture
- Optimized for mobile and low-computation environments
- Used as primary baseline model

### 🔹 ResNet50 (Comparative Model)
- Deep residual learning architecture
- Improved feature extraction capability
- Used for performance benchmarking

---

## 📊 Dataset Description

- **Source**: Kaggle Chest X-ray Pneumonia Dataset
- **Classes**:
  - Normal
  - Pneumonia

### Dataset Variants:
- Full dataset (original distribution)
- Reduced dataset (~1500–2000 images) for efficiency analysis

---

## 🔬 Experimental Setup

All experiments are conducted under controlled conditions:

| Parameter | Value |
|----------|------|
| Input Size | 224 × 224 |
| Batch Size | 16 |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Epochs | 5–10 |
| Framework | TensorFlow / Keras |
| Hardware | Google Colab GPU |

---

## 📈 Evaluation Metrics

The model performance is evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-score
- Confusion Matrix
- ROC-AUC Curve

---

## 🔥 Explainability: Grad-CAM

To ensure clinical interpretability, **Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to visualize model attention.

### Grad-CAM Analysis Includes:
- Correct predictions (true positives)
- Misclassified cases (false positives / false negatives)
- Heatmap overlays on chest X-ray images

### Objective:
To verify whether the model focuses on **lung regions** or irrelevant anatomical structures.

---

## 📉 Error Analysis

We perform qualitative and quantitative error analysis:

### False Positives:
- Normal images incorrectly classified as Pneumonia
- Often caused by:
  - Image artifacts
  - Rib shadows
  - Low contrast regions

### False Negatives:
- Pneumonia cases misclassified as Normal
- Often caused by:
  - Mild infections
  - Poor image quality
  - Subtle radiographic patterns

---

## ⚖️ Dataset Efficiency Study

To evaluate real-world applicability, we compare:

- Full dataset performance
- Reduced dataset performance

### Objective:
Assess model robustness under limited data conditions.

---

## 📦 Installation

```bash
git clone https://github.com/your-username/MedX-PneuNet.git
cd MedX-PneuNet

pip install -r requirements.txt
````

---

## 🚀 Training the Model

### MobileNetV2

```bash
python experiments/exp_mobilenet_full.py
```

### ResNet50

```bash
python experiments/exp_resnet_full.py
```

---

## 📊 Results Summary (Example Format)

| Model       | Dataset | Accuracy | F1-score | ROC-AUC |
| ----------- | ------- | -------- | -------- | ------- |
| MobileNetV2 | Full    | XX%      | XX%      | XX%     |
| MobileNetV2 | Reduced | XX%      | XX%      | XX%     |
| ResNet50    | Full    | XX%      | XX%      | XX%     |
| ResNet50    | Reduced | XX%      | XX%      | XX%     |

---

## 🧪 Grad-CAM Visualization

Grad-CAM outputs highlight discriminative regions in chest X-ray images, enabling interpretability of model predictions.

Example outputs include:

* Pneumonia region localization
* Lung abnormality detection
* Misclassification reasoning

---

## 🌐 Streamlit Deployment (Optional)

Run the web application:

```bash
streamlit run streamlit_app/app.py
```

Features:

* Upload chest X-ray image
* Predict Pneumonia / Normal
* Display confidence score
* Visualize Grad-CAM heatmap

---

## 📚 Project Structure

```
MedX-PneuNet/
│
├── src/                  # Core implementation
├── experiments/         # Model training scripts
├── evaluation/          # Metrics & analysis
├── explainability/      # Grad-CAM implementation
├── streamlit_app/       # Web deployment
├── results/             # Outputs & visualizations
├── data/                # Dataset (local use only)
├── notebooks/           # Exploratory analysis
└── README.md
```

---

## 🔮 Future Work

* Integration of Vision Transformers (ViTs)
* Multi-class lung disease classification
* Clinical validation with real hospital data
* Lightweight mobile deployment (Android/iOS)
* Uncertainty quantification for predictions

---

## 📖 Citation

If you use this repository in your research, please cite:

```
MedX-PneuNet: Explainable Deep Learning for Pneumonia Detection using Chest X-ray Images
Author: [Your Name]
Year: 2026
```

---

## 📜 License

This project is licensed under the MIT License.

---

## 🤝 Acknowledgements

* Kaggle Chest X-ray dataset contributors
* Original research paper authors
* TensorFlow and Keras communities

---

## ⭐ Final Note

This project is developed for **academic research and educational purposes**, focusing on **explainable AI in medical imaging**.