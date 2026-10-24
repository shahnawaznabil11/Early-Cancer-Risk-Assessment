# 🫁 Early Pulmonary Risk Assessment Platform

An end-to-end clinical decision-support web application that evaluates pulmonary malignancy risk using observational clinical markers and a balanced Random Forest classifier.

🔗 **Live Application:** [early-cancer-risk-assessment.streamlit.app](https://early-cancer-risk-assessment.streamlit.app)

---

## 📌 Key Features

- **Multi-Step Clinical Wizard:** Interactive, questionnaire-driven interface with collapsible clinical explanations for each queried symptom.
- **Dynamic Risk Stratification:** Instant statistical risk inference and natural-frequency cohort visualization (e.g., *X out of 100 individuals*).
- **Interactive Counterfactual Simulation:** Real-time toggles to simulate symptom resolution and examine score adjustments without re-entering data.
- **Clinical PDF Export:** On-the-fly downloadable medical summary reports generated via ReportLab for physician consultations.

---

## 🛠️ Tech Stack

- **Frontend & App Framework:** Streamlit
- **Machine Learning Engine:** Scikit-learn, Joblib
- **Data Manipulation:** Pandas, NumPy
- **Document Generation:** ReportLab
- **Cloud Hosting:** Streamlit Community Cloud

---

## 🚀 Local Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shahnawaznabil11/Early-Cancer-Risk-Assessment.git](https://github.com/shahnawaznabil11/Early-Cancer-Risk-Assessment.git)
   cd Early-Cancer-Risk-Assessment
