"""
Traffic Violation Detection System — Streamlit Frontend
Loads DNN, CV (YOLOv8s), NLP (mBERT) models + RAG pipeline from Hugging Face Hub.
"""

import streamlit as st
import numpy as np
import pickle
import torch
import os
from PIL import Image
from datetime import datetime, timedelta
import random

from huggingface_hub import hf_hub_download, snapshot_download

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
HF_REPO = "zainab-abid/traffic-violation-models"

st.set_page_config(
    page_title="Traffic Violation Detection System",
    page_icon="🚦",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached — only runs once per session)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading DNN model...")
def load_dnn_model():
    import tf_keras as keras

    model_path = hf_hub_download(repo_id=HF_REPO, filename="dnn_model.keras")
    scaler_path = hf_hub_download(repo_id=HF_REPO, filename="scaler.pkl")
    encoder_path = hf_hub_download(repo_id=HF_REPO, filename="label_encoder.pkl")

    model = keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(encoder_path, "rb") as f:
        le_target = pickle.load(f)

    return model, scaler, le_target


@st.cache_resource(show_spinner="Loading CV (YOLOv8s) model...")
def load_cv_model():
    from ultralytics import YOLO

    model_path = hf_hub_download(repo_id=HF_REPO, filename="cv_model_yolov8s_best.pt")
    bundle_path = hf_hub_download(repo_id=HF_REPO, filename="cv_bundle.pkl")

    cv_model = YOLO(model_path)
    with open(bundle_path, "rb") as f:
        cv_bundle = pickle.load(f)

    return cv_model, cv_bundle


@st.cache_resource(show_spinner="Loading NLP (mBERT) model...")
def load_nlp_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Downloads the whole nlp_model/ folder from the repo
    nlp_dir = snapshot_download(repo_id=HF_REPO, allow_patterns=["nlp_model/*"])
    nlp_model_path = os.path.join(nlp_dir, "nlp_model")

    tokenizer = AutoTokenizer.from_pretrained(nlp_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(nlp_model_path)
    model.eval()

    encoder_path = hf_hub_download(repo_id=HF_REPO, filename="nlp_label_encoder.pkl")
    with open(encoder_path, "rb") as f:
        le_nlp = pickle.load(f)

    return model, tokenizer, le_nlp


# ──────────────────────────────────────────────────────────────────────────────
# RAG QUERY FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
# ── Simplified RAG — direct lookup dictionary (temporary, FAISS-free) ────────
VIOLATION_LAW_DB = {
    "Citation": {
        "legal_ref": "Motor Vehicle Ordinance 1965, Section 139",
        "fine": 5000,
        "description": "Vehicle was found in serious violation of traffic rules.",
    },
    "Warning": {
        "legal_ref": "Motor Vehicle Ordinance 1965, Section 85",
        "fine": 1000,
        "description": "Vehicle was found in minor violation of traffic rules.",
    },
    "ESERO": {
        "legal_ref": "Motor Vehicle Ordinance 1965, Section 102",
        "fine": 2000,
        "description": "Rider found without helmet or driving in wrong lane.",
    },
    "SERO": {
        "legal_ref": "Motor Vehicle Ordinance 1965, Section 117",
        "fine": 3000,
        "description": "Vehicle detected crossing a red traffic signal.",
    },
    "No Helmet": {
        "legal_ref": "Motor Vehicle Ordinance 1965, Section 89-A",
        "fine": 2000,
        "description": "Rider operating motorcycle without wearing a safety helmet.",
    },
    "Wrong Lane": {
        "legal_ref": "Motor Vehicle Ordinance 1965, Section 109",
        "fine": 2500,
        "description": "Vehicle found driving in the wrong lane against traffic.",
    },
    "Red Light": {
        "legal_ref": "Motor Vehicle Ordinance 1965, Section 117",
        "fine": 3000,
        "description": "Vehicle detected crossing a red traffic signal without stopping.",
    },
}

def query_law(violation_type: str) -> dict:
    """Direct dictionary lookup — no FAISS/embeddings needed."""
    info = VIOLATION_LAW_DB.get(violation_type, {
        "legal_ref": "Motor Vehicle Ordinance 1965 (General)",
        "fine": 1000,
        "description": "Traffic violation recorded.",
    })
    return {
        "violation_type": violation_type,
        "legal_ref": info["legal_ref"],
        "fine": info["fine"],
        "description": info["description"],
    }

# ──────────────────────────────────────────────────────────────────────────────
# CHALLAN PDF GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_challan_pdf(violation_type, confidence, name="Unknown",
                          plate="XXX-0000", city="Lahore", officer="Officer On Duty"):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    import tempfile

    rag_info = query_law(violation_type)

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(2 * cm, height - 2 * cm, "TRAFFIC CHALLAN NOTICE")

    c.setFont("Helvetica", 11)
    lines = [
        f"Challan No: TP-{random.randint(10000, 99999)}",
        f"Issuing Authority: Traffic Police, {city}",
        f"Name: {name}",
        f"Vehicle: {plate}",
        f"Date: {datetime.now().strftime('%d-%m-%Y')}",
        f"Court Date: {(datetime.now() + timedelta(days=30)).strftime('%d-%m-%Y')}",
        f"Issuing Officer: {officer}",
        "",
        f"Violation Type: {violation_type}",
        f"Detection Confidence: {confidence:.1%}",
        f"Legal Reference: {rag_info['legal_ref']}",
        f"Fine Amount: PKR {rag_info['fine']:,}",
        "",
        f"Details: {rag_info['description']}",
    ]

    y = height - 3.5 * cm
    for line in lines:
        c.drawString(2 * cm, y, line)
        y -= 0.7 * cm

    c.save()
    return path


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def predict_dnn(feature_row):
    model, scaler, le_target = load_dnn_model()
    sample = scaler.transform(np.array(feature_row).reshape(1, -1))
    probs = model.predict(sample, verbose=0)
    pred = np.argmax(probs, axis=1)[0]
    conf = float(np.max(probs, axis=1)[0])
    v_type = le_target.inverse_transform([pred])[0]
    return v_type, conf


def predict_cv(image_path, conf_threshold=0.35):
    cv_model, cv_bundle = load_cv_model()
    class_names = cv_bundle["class_names"]
    class_map = cv_bundle["class_map"]

    results = cv_model(image_path, verbose=False)[0]
    detections = []
    for box in results.boxes:
        conf = float(box.conf)
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls)
        cls_name = class_names[cls_id] if cls_id < len(class_names) else "unknown"
        xyxy = box.xyxy[0].tolist()
        detections.append({
            "violation_type": class_map.get(cls_name, cls_name),
            "confidence": round(conf, 4),
            "bbox": [round(v, 1) for v in xyxy],
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    if not detections:
        return None, 0.0, []
    return detections[0]["violation_type"], detections[0]["confidence"], detections


def predict_nlp(text):
    model, tokenizer, le_nlp = load_nlp_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                        padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    pred_idx = probs.argmax()
    v_type = le_nlp.inverse_transform([pred_idx])[0]
    conf = float(probs[pred_idx])
    return v_type, conf


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

st.title("🚦 Traffic Violation Detection System")
st.caption("DNN + CV (YOLOv8s) + NLP (mBERT) → RAG → Auto-generated Challan")

tab_cv, tab_nlp, tab_dnn = st.tabs(["📷 Image Detection (CV)", "📝 Text Description (NLP)", "📊 Tabular Data (DNN)"])

# ── CV TAB ─────────────────────────────────────────────────────────────────────
with tab_cv:
    st.subheader("Upload a traffic camera image")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="cv_upload")

    col1, col2 = st.columns(2)
    with col1:
        cv_name = st.text_input("Driver name", "Unknown", key="cv_name")
        cv_plate = st.text_input("Vehicle plate", "XXX-0000", key="cv_plate")
    with col2:
        cv_city = st.selectbox("City", ["Lahore", "Karachi", "Islamabad", "Rawalpindi",
                                         "Faisalabad", "Multan"], key="cv_city")

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded image")

        if st.button("Detect Violation", key="cv_detect"):
            with st.spinner("Running detection..."):
                import tempfile
                # Windows-safe temp file (auto-detects correct temp directory)
                suffix = os.path.splitext(uploaded_image.name)[1]
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp_file.write(uploaded_image.getbuffer())
                tmp_file.close()
                tmp_path = tmp_file.name
                v_type, conf, detections = predict_cv(tmp_path)

            if v_type is None:
                st.success("✅ No violation detected in this image.")
            else:
                st.error(f"🚨 Violation detected: **{v_type}** ({conf:.1%} confidence)")

                rag_info = query_law(v_type)
                st.write(f"**Legal Reference:** {rag_info['legal_ref']}")
                st.write(f"**Fine Amount:** PKR {rag_info['fine']:,}")

                pdf_path = generate_challan_pdf(v_type, conf, cv_name, cv_plate, cv_city)
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download Challan PDF", f, file_name="challan.pdf",
                                        mime="application/pdf")

# ── NLP TAB ────────────────────────────────────────────────────────────────────
with tab_nlp:
    st.subheader("Describe the violation (English or Urdu)")
    nlp_text = st.text_area("Violation description", height=120, key="nlp_text",
                             placeholder="e.g. Rider was found riding without a helmet...")

    col1, col2 = st.columns(2)
    with col1:
        nlp_name = st.text_input("Driver name", "Unknown", key="nlp_name")
        nlp_plate = st.text_input("Vehicle plate", "XXX-0000", key="nlp_plate")
    with col2:
        nlp_city = st.selectbox("City", ["Lahore", "Karachi", "Islamabad", "Rawalpindi",
                                          "Faisalabad", "Multan"], key="nlp_city")

    if st.button("Classify Violation", key="nlp_detect"):
        if not nlp_text.strip():
            st.warning("Please enter a description first.")
        else:
            with st.spinner("Classifying..."):
                v_type, conf = predict_nlp(nlp_text)

            st.error(f"🚨 Classified as: **{v_type}** ({conf:.1%} confidence)")

            rag_info = query_law(v_type)
            st.write(f"**Legal Reference:** {rag_info['legal_ref']}")
            st.write(f"**Fine Amount:** PKR {rag_info['fine']:,}")

            pdf_path = generate_challan_pdf(v_type, conf, nlp_name, nlp_plate, nlp_city)
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download Challan PDF", f, file_name="challan.pdf",
                                    mime="application/pdf")

# ── DNN TAB ────────────────────────────────────────────────────────────────────
with tab_dnn:
    st.subheader("Enter tabular violation details")
    st.caption("Fill in the fields matching your training feature set.")

    # NOTE: adjust these inputs to match the EXACT features your DNN was trained on
    col1, col2, col3 = st.columns(3)
    with col1:
        speed = st.number_input("Speed (km/h)", 0, 200, 60)
        hour = st.number_input("Hour of day (0-23)", 0, 23, 14)
    with col2:
        weather_idx = st.selectbox("Weather", ["Clear", "Rain", "Fog"], key="dnn_weather")
        road_idx = st.selectbox("Road type", ["Highway", "City", "Residential"], key="dnn_road")
    with col3:
        dnn_name = st.text_input("Driver name", "Unknown", key="dnn_name")
        dnn_plate = st.text_input("Vehicle plate", "XXX-0000", key="dnn_plate")

    if st.button("Predict Violation", key="dnn_detect"):
        # Placeholder feature vector — REPLACE with your actual feature engineering
        feature_row = [speed, hour, 0, 0]  # adjust length/order to match scaler

        with st.spinner("Predicting..."):
            v_type, conf = predict_dnn(feature_row)

        st.error(f"🚨 Predicted: **{v_type}** ({conf:.1%} confidence)")

        rag_info = query_law(v_type)
        st.write(f"**Legal Reference:** {rag_info['legal_ref']}")
        st.write(f"**Fine Amount:** PKR {rag_info['fine']:,}")

        pdf_path = generate_challan_pdf(v_type, conf, dnn_name, dnn_plate, "Lahore")
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Challan PDF", f, file_name="challan.pdf",
                                mime="application/pdf")

st.divider()
st.caption("Traffic Violation Detection System — DNN + CV + NLP + RAG pipeline")
