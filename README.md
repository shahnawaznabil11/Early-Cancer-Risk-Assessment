# 🩺 Multi-Modal Cancer Risk Assessment & Diagnostic Platform

An end-to-end clinical decision-support application combining tabular pulmonary cancer risk stratification with deep learning-based Brain MRI tumor classification and explainability.

---

## 🚀 Key Modules

### 1. 🫁 Pulmonary Risk Assessment
- **Clinical Questionnaire Wizard**: Multi-step survey capturing demographic markers and observational symptoms.
- **Dynamic Risk Stratification**: Real-time risk probability calculation and cohort natural-frequency dot visualization.
- **Clinical PDF Export**: Automated diagnostic summary report generation via ReportLab for physician consultation.

### 2. 🧠 Brain MRI Tumor Classification & Explainability
- **Deep Learning Vision Classifier**: PyTorch ResNet-18 architecture classifying axial brain MRI scans into 4 target classes:
  - *Glioma Tumor*
  - *Meningioma Tumor*
  - *Pituitary Tumor*
  - *No Tumor*
- **Grad-CAM Visual Explanations**: Gradient-weighted Class Activation Mapping overlay highlighting focal anatomical regions driving the model's prediction.
- **Probability Breakdown**: Full multi-class softmax probability distribution.

---

## 🛠️ Tech Stack & Dependencies
- **Frontend / Application Engine**: Streamlit
- **Machine Learning**: Scikit-Learn, PyTorch, Torchvision
- **Explainability**: Grad-CAM, SHAP
- **Image Processing & Data Handling**: OpenCV (`opencv-python-headless`), Pillow, NumPy, Pandas
- **Reporting**: ReportLab

---

## 📦 Setup & Local Execution

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/shahnawaznabil11/Early-Cancer-Risk-Assessment.git](https://github.com/shahnawaznabil11/Early-Cancer-Risk-Assessment.git)
   cd Early-Cancer-Risk-Assessment
