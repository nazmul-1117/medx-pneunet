# 📘 MedX-PneuNet

## 🫁 Explainable Deep Learning for Pneumonia Detection from Chest X-ray Images Using Lightweight CNNs and Grad-CAM

---

## 📌 Overview

**MedX-PneuNet** is a deep learning-based medical image classification system designed to detect **Pneumonia from Chest X-ray images** using lightweight CNN architectures. The project focuses not only on accuracy but also on **model interpretability** using **Grad-CAM explainability techniques**.

It compares multiple models and evaluates their performance under different dataset conditions to analyze real-world applicability in low-resource environments.

---

## 🎯 Objectives

* Detect Pneumonia from chest X-ray images
* Compare lightweight and deep CNN architectures
* Evaluate performance on full vs reduced datasets
* Provide explainable AI visualizations using Grad-CAM
* Improve clinical interpretability of deep learning models

---

## 🧠 Models Used

* **MobileNetV2** (Lightweight baseline model)
* **ResNet50** (Deeper architecture for comparison)

Both models are fine-tuned for binary classification:

* Normal
* Pneumonia

---

## 📊 Dataset

We use the publicly available dataset:

Chest X-Ray Images (Pneumonia)

### Dataset Split:

* Training set: 5216 images
* Validation set: 16 images
* Test set: 624 images

### Classes:

* NORMAL
* PNEUMONIA

---

## ⚙️ Project Pipeline

1. Data Loading & Preprocessing
2. Model Training (MobileNetV2 & ResNet50)
3. Performance Evaluation
4. Dataset Efficiency Experiment (Full vs Reduced)
5. Explainability using Grad-CAM
6. Error Analysis (False Positives / False Negatives)

---

## 📈 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC (optional)

---

## 🔍 Explainability (Grad-CAM)

Grad-CAM is used to visualize model attention on chest X-ray images.

### Key Insights:

* Highlights lung regions influencing predictions
* Detects whether model focuses on irrelevant areas
* Compares attention maps between MobileNetV2 and ResNet50
* Helps interpret false predictions

---

## 🧪 Experimental Analysis

### 1. Model Comparison

* MobileNetV2 vs ResNet50 performance comparison

### 2. Dataset Efficiency

* Full dataset training
* Reduced dataset (~1500–2000 images)
* Analysis of performance drop

### 3. Error Analysis

* False Positives: Normal classified as Pneumonia
* False Negatives: Missed Pneumonia cases
* Causes: noise, low contrast, feature limitations

---

## 🧰 Tech Stack

* Python
* TensorFlow / Keras
* NumPy / Pandas
* Matplotlib / Seaborn
* OpenCV
* Scikit-learn

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-username/medx-pneunet.git
cd medx-pneunet
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run Training Notebook

Open:

```
notebooks/02_mobilenetv2_training.ipynb
```

or

```
notebooks/03_resnet50_training.ipynb
```

---

### 4. Run Evaluation

```bash
python src/evaluate.py
```

---

### 5. Run Grad-CAM

```bash
python src/gradcam.py
```

---

## 📊 Results (Example Format)

| Model       | Accuracy | Precision | Recall | F1-score |
| ----------- | -------- | --------- | ------ | -------- |
| MobileNetV2 | XX%      | XX%       | XX%    | XX%      |
| ResNet50    | XX%      | XX%       | XX%    | XX%      |

---

## 🔬 Key Contributions

* Lightweight CNN comparison for medical imaging
* Dataset efficiency study for low-resource settings
* Explainable AI integration using Grad-CAM
* Error analysis for clinical interpretability
* CPU-friendly training pipeline

---

## 🖼️ Grad-CAM Visualization

(Insert sample images here)

* Correct prediction heatmaps
* Incorrect prediction analysis
* Model attention comparison

---

## 📌 Future Work

* Improve dataset balancing
* Integrate transformer-based models
* Deploy as a web application (Streamlit)
* Add multi-disease classification (COVID-19, TB, etc.)

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**MedX-PneuNet Project**
Final Year Academic Research Project
Focus: Deep Learning + Medical Imaging + Explainable AI

---

## ⭐ Acknowledgements

* Kaggle dataset contributors
* TensorFlow/Keras community
* Research in explainable AI (Grad-CAM)