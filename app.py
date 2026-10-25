import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image
from gradcam_utils import load_brain_model, generate_superimposed_mri

import os
import urllib.request
# Ensure brain model weights exist; download from GitHub Release if missing
MODEL_PATH = "brain_mri_model.pth"
MODEL_URL = "https://github.com/shahnawaznabil11/Early-Cancer-Risk-Assessment/releases/download/v1.0.0/brain_mri_model.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading brain MRI model weights (one-time setup)..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
# =====================================================================
# 1. Page Configuration & Navigation Tabs
# =====================================================================
st.set_page_config(
    page_title="Multi-Modal Oncology Clinical Assistant",
    page_icon="🏥",
    layout="wide"
)

tab1, tab2 = st.tabs(["🫁 Pulmonary Risk Assessment", "🧠 Brain MRI Tumor Classification"])

# =====================================================================
# TAB 1: Pulmonary Risk Assessment (Your Complete Original Engine)
# =====================================================================
with tab1:
    st.markdown("""
    <style>
        .big-stat {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .traffic-banner {
            padding: 1.2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        .traffic-green {
            background-color: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #16a34a;
        }
        .traffic-yellow {
            background-color: rgba(234, 179, 8, 0.12);
            border: 1px solid rgba(234, 179, 8, 0.4);
            color: #ca8a04;
        }
        .traffic-red {
            background-color: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #dc2626;
        }
        .array-container {
            font-size: 1.25rem;
            line-height: 1.6;
            letter-spacing: 0.15rem;
            word-break: break-all;
            background: rgba(125, 125, 125, 0.05);
            padding: 1rem;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. Artifact Loader
    # -------------------------------------------------------------
    @st.cache_resource
    def load_artifacts():
        model = joblib.load("cancer_risk_model.pkl")
        features = joblib.load("model_features.pkl")
        return model, features

    try:
        model, model_features = load_artifacts()
    except Exception as e:
        st.error(f"Could not load artifacts: {e}. Check that cancer_risk_model.pkl and model_features.pkl exist.")
        st.stop()

    # -------------------------------------------------------------
    # 3. Clinical Questions Configuration
    # -------------------------------------------------------------
    QUESTIONS = [
        {
            "id": "GENDER",
            "title": "What was your biological sex assigned at birth?",
            "type": "select",
            "options": ["Male", "Female"],
            "why": "Biological sex influences pulmonary physiology, hormonal modulation, and differing baseline lung cancer susceptibilities across epidemiological cohorts."
        },
        {
            "id": "AGE",
            "title": "What is your current age?",
            "type": "slider",
            "min": 18,
            "max": 95,
            "default": 60,
            "why": "Age is one of the strongest epidemiological determinants for thoracic malignancies due to cumulative cellular exposure and biological senescence."
        },
        {
            "id": "SMOKING",
            "title": "Do you currently smoke or have a history of smoking tobacco?",
            "type": "binary",
            "why": "Tobacco combustion introduces direct carcinogens that damage bronchial DNA and drive high long-term pulmonary risk."
        },
        {
            "id": "YELLOW_FINGERS",
            "title": "Have you noticed yellow nicotine staining on your fingers?",
            "type": "binary",
            "why": "Yellow finger staining serves as a strong biological marker of sustained, heavy cigarette handling and chronic tar exposure."
        },
        {
            "id": "ANXIETY",
            "title": "Do you regularly experience persistent anxiety or tension?",
            "type": "binary",
            "why": "Chronic psychological distress and autonomic stress responses frequently co-occur with altered somatic symptoms and behavioral lifestyle habits."
        },
        {
            "id": "PEER_PRESSURE",
            "title": "Have social or workplace environments encouraged smoking behaviors?",
            "type": "binary",
            "why": "Social contagion and environmental exposure heavily reinforce recurring tobacco habits and secondhand carcinogen inhalation."
        },
        {
            "id": "CHRONIC_DISEASE",
            "title": "Have you been diagnosed with any chronic health conditions?",
            "type": "binary",
            "why": "Underlying chronic inflammatory and metabolic disorders can elevate vulnerability to systemic pulmonary pathology."
        },
        {
            "id": "FATIGUE",
            "title": "Do you experience ongoing, unexplained fatigue not relieved by rest?",
            "type": "binary",
            "why": "Systemic malignant processes demand substantial metabolic resources, often presenting early as chronic or progressive lethargy."
        },
        {
            "id": "ALLERGY",
            "title": "Do you suffer from frequent allergies or hyper-reactive airways?",
            "type": "binary",
            "why": "Allergic responses illuminate airway hyper-reactivity and immune baseline activity within mucosal respiratory tissues."
        },
        {
            "id": "WHEEZING",
            "title": "Do you experience high-pitched wheezing sounds when exhaling?",
            "type": "binary",
            "why": "Wheezing indicates partial airway narrowing, bronchial constriction, or localized obstruction in the respiratory tract."
        },
        {
            "id": "ALCOHOL_CONSUMING",
            "title": "Do you consume alcohol regularly or in moderate-to-high amounts?",
            "type": "binary",
            "why": "Alcohol functions as an irritant and potential co-carcinogen, compounding the cellular burden placed on mucosal tissue."
        },
        {
            "id": "COUGHING",
            "title": "Have you had a persistent cough that does not clear?",
            "type": "binary",
            "why": "A stubborn, unresolving cough is one of the classic early primary warning signs of endobronchial irritation or tumor progression."
        },
        {
            "id": "SHORTNESS_OF_BREATH",
            "title": "Do you experience breathlessness or shortness of breath during daily routines?",
            "type": "binary",
            "why": "Dyspnea occurs when compromised gas exchange or localized airway occlusion restricts optimal alveolar ventilation."
        },
        {
            "id": "SWALLOWING_DIFFICULTY",
            "title": "Do you experience painful or difficult swallowing (dysphagia)?",
            "type": "binary",
            "why": "Difficulty swallowing may indicate localized thoracic expansion or anatomical compression along mediastinal structures."
        },
        {
            "id": "CHEST_PAIN",
            "title": "Do you experience persistent, localized, or deep chest pain?",
            "type": "binary",
            "why": "Intrathoracic pain frequently points to parietal pleural irritation, nerve involvement, or chronic pulmonary inflammation."
        }
    ]

    # -------------------------------------------------------------
    # 4. Helpers & PDF Generator
    # -------------------------------------------------------------
    if "step" not in st.session_state:
        st.session_state.step = 0

    if "responses" not in st.session_state:
        st.session_state.responses = {}

    def get_current_model_input(base_dict):
        formatted = {}
        for col in model_features:
            if col == "GENDER":
                val = base_dict.get("GENDER", "Female")
                formatted[col] = 1 if val == "Male" else 0
            elif col == "AGE":
                formatted[col] = int(base_dict.get("AGE", 60))
            else:
                val = base_dict.get(col, "No")
                formatted[col] = 1 if val == "Yes" else 0
        return pd.DataFrame([formatted])[model_features]

    def generate_pdf(responses, count, category):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"))
        meta_style = ParagraphStyle('M', parent=styles['Normal'], fontSize=9, leading=12, textColor=colors.HexColor("#64748b"))
        head_style = ParagraphStyle('H', parent=styles['Heading2'], fontSize=12, leading=16, textColor=colors.HexColor("#1e293b"))
        disc_style = ParagraphStyle('D', parent=styles['Normal'], fontSize=8, leading=11, textColor=colors.HexColor("#94a3b8"))

        story.append(Paragraph("Clinical Decision-Support: Pulmonary Risk Summary", title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y - %H:%M')} | Tool: PulmoCheck AI", meta_style))
        story.append(Spacer(1, 12))

        summary_data = [
            ["Patient Age / Sex", f"{responses.get('AGE', 'N/A')} yrs / {responses.get('GENDER', 'N/A')}"],
            ["Statistical Probability", f"{count}%"],
            ["Risk Stratification", category.upper()],
            ["Cohort Context", f"{count} out of 100 people with similar profile were positive"]
        ]
        t1 = Table(summary_data, colWidths=[180, 330])
        t1.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('PADDING', (0, 0), (-1, -1), 6)
        ]))
        story.append(t1)
        story.append(Spacer(1, 14))

        story.append(Paragraph("Reported Factors", head_style))
        sym_rows = [["Factor", "Status"]]
        for k, v in responses.items():
            if k not in ['GENDER', 'AGE']:
                sym_rows.append([k.replace('_', ' ').title(), "POSITIVE" if v == "Yes" else "Negative"])
        t2 = Table(sym_rows, colWidths=[350, 160])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 5)
        ]))
        story.append(t2)
        story.append(Spacer(1, 14))

        story.append(Paragraph(
            "Notice: This document provides educational risk stratification from a machine learning model. "
            "It does not replace clinical consultation or radiological tests (e.g., Low-Dose CT).", disc_style
        ))
        doc.build(story)
        buffer.seek(0)
        return buffer

    # -------------------------------------------------------------
    # 5. Screen 0: Welcome Screen
    # -------------------------------------------------------------
    if st.session_state.step == 0:
        st.title("🫁 Pulmonary Health Assessment")
        st.markdown("### Take 2 minutes to evaluate your clinical pulmonary risk profile.")
        st.markdown("This tool uses a machine learning model trained on clinical markers to highlight preliminary risk factors.")
        st.info("**Not a Diagnosis**: This tool calculates statistical probabilities from observational data and does not replace medical screening or professional clinical advice.")
        st.write("")
        if st.button("Start Assessment ➔", type="primary", use_container_width=True):
            st.session_state.step = 1
            st.rerun()

    # -------------------------------------------------------------
    # 6. Screens 1 to 15: Wizard Questions
    # -------------------------------------------------------------
    elif 1 <= st.session_state.step <= len(QUESTIONS):
        q_idx = st.session_state.step - 1
        q = QUESTIONS[q_idx]

        st.progress(st.session_state.step / len(QUESTIONS))
        st.caption(f"Question {st.session_state.step} of {len(QUESTIONS)}")

        st.subheader(q["title"])
        current_val = st.session_state.responses.get(q["id"])

        if q["type"] == "binary":
            default_idx = 0 if current_val == "No" else (1 if current_val == "Yes" else 0)
            answer = st.radio("Select an answer:", ["No", "Yes"], index=default_idx, horizontal=True)
        elif q["type"] == "select":
            default_idx = 0 if current_val == "Male" else (1 if current_val == "Female" else 0)
            answer = st.radio("Select your sex:", q["options"], index=default_idx, horizontal=True)
        elif q["type"] == "slider":
            default_age = current_val if current_val is not None else q["default"]
            answer = st.slider("Select age:", min_value=q["min"], max_value=q["max"], value=default_age)

        with st.expander("🔍 Why are we asking this?"):
            st.write(q["why"])

        st.write("---")
        nav1, nav2 = st.columns(2)
        with nav1:
            if st.button("⬅ Back", use_container_width=True):
                st.session_state.responses[q["id"]] = answer
                st.session_state.step -= 1
                st.rerun()
        with nav2:
            btn_txt = "Finish & See Results ➔" if st.session_state.step == len(QUESTIONS) else "Next ➔"
            if st.button(btn_txt, type="primary", use_container_width=True):
                st.session_state.responses[q["id"]] = answer
                st.session_state.step += 1
                st.rerun()

    # -------------------------------------------------------------
    # 7. Screen 16: Results & Export
    # -------------------------------------------------------------
    elif st.session_state.step > len(QUESTIONS):
        st.title("Assessment Findings")

        user_df = get_current_model_input(st.session_state.responses)
        prob = model.predict_proba(user_df)[0][1]
        people_count = int(round(prob * 100))

        if prob >= 0.70:
            headline_class, headline_title, headline_desc = "traffic-red", "High Risk Identified", "Clinical presentation aligns with high-frequency risk cohorts."
        elif prob >= 0.40:
            headline_class, headline_yellow, headline_desc = "traffic-yellow", "Moderate / Elevated Risk", "Noticeable accumulation of clinical pulmonary risk flags."
        else:
            headline_class, headline_title, headline_desc = "traffic-green", "Low Risk Profile", "Few reported clinical indicators corresponding with malignancy."

        st.markdown(f"""
        <div class="traffic-banner {headline_class}">
            <div class="big-stat">{headline_title}</div>
            <div>{headline_desc}</div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Statistical Cohort Visualization")
        st.write("Out of 100 people presenting with your exact demographic and symptom profile:")
        
        green_dots = "".join(["🟢 " for _ in range(100 - people_count)])
        red_dots = "".join(["🔴 " for _ in range(people_count)])
        st.markdown(f'<div class="array-container">{red_dots}{green_dots}</div>', unsafe_allow_html=True)
        st.caption(f"🔴 **{people_count} individuals** were diagnosed with lung cancer | 🟢 **{100 - people_count} individuals** were not.")

        st.divider()
        st.subheader("Factor Breakdown")
        reported_positive = [k for k, v in st.session_state.responses.items() if v == "Yes"]
        reported_negative = [k for k, v in st.session_state.responses.items() if v == "No"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### ⚠️ Pushed Score Up")
            if reported_positive:
                for item in reported_positive:
                    st.write(f"- {item.replace('_', ' ').title()}")
            else:
                st.write("*No positive lifestyle/symptom flags reported.*")

        with col2:
            st.markdown("##### 🛡️ Pushed Score Down")
            if reported_negative:
                for item in reported_negative:
                    st.write(f"- {item.replace('_', ' ').title()}")
            else:
                st.write("*No protective absence flags.*")

        st.divider()
        pdf_buffer = generate_pdf(st.session_state.responses, people_count, headline_title)
        st.download_button(
            label="📄 Download Doctor's Summary Report (PDF)",
            data=pdf_buffer,
            file_name="pulmonary_risk_summary.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.write("")
        st.error("🚨 **Emergency Notice**: If experiencing severe chest pain, extreme breathlessness, or coughing up blood, seek immediate emergency care.")

        st.write("")
        if st.button("↺ Start Over / Test New Patient", use_container_width=True):
            st.session_state.step = 0
            st.session_state.responses = {}
            st.rerun()


# =====================================================================
# TAB 2: Brain MRI Scanner (ResNet18 + Grad-CAM Heatmaps)
# =====================================================================
with tab2:
    st.header("Brain MRI Tumor Detection & Explainability")
    st.caption("Upload an axial MRI brain scan to classify tumor presence and view Grad-CAM feature heatmaps.")

    @st.cache_resource
    def get_brain_model():
        return load_brain_model("brain_mri_model.pth")

    try:
        brain_model, class_names = get_brain_model()
        model_loaded = True
    except Exception as err:
        st.error(f"Failed to load MRI model: {err}")
        model_loaded = False

    if model_loaded:
        uploaded_file = st.file_uploader(
            "Upload Brain MRI Slice (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="brain_mri_uploader"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            with st.spinner("Processing scan and computing Grad-CAM activations..."):
                results = generate_superimposed_mri(image, brain_model, class_names)

            clean_pred = results["prediction"].replace("_", " ").title()
            confidence = results["confidence"] * 100

            st.write("---")
            col_mri1, col_mri2 = st.columns(2)

            with col_mri1:
                st.subheader("Original MRI Slice")
                st.image(image, use_container_width=True)

            with col_mri2:
                st.subheader("Grad-CAM Visual Explanation")
                st.image(results["superimposed_image"], use_container_width=True)
                st.caption("Warmer regions (red/yellow) indicate focal features influencing ResNet18.")

            st.write("---")
            if "No Tumor" in clean_pred:
                st.success(f"### Assessment: {clean_pred} (Confidence: {confidence:.2f}%)")
            else:
                st.error(f"### Assessment: {clean_pred} Detected (Confidence: {confidence:.2f}%)")

            with st.expander("Detailed Probability Distribution"):
                for tumor_type, prob in results["all_probabilities"].items():
                    name_display = tumor_type.replace("_", " ").title()
                    st.write(f"**{name_display}**: `{prob * 100:.2f}%`")
                    st.progress(float(prob))
