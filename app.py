import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# App title and tagline (centered and styled)
st.set_page_config(page_title="MEDISCAN", layout="centered")

st.markdown(
    """
    <div style='text-align: center; margin-top: -40px; margin-bottom: 0px;'>
        <h1 style='color: #00FA9A; font-size: 3.2em; margin-bottom: 0.2em;'>MEDISCAN</h1>
        <p style='font-size: 1.3em; font-style: italic; color: #00BFFF; margin-top: 0.1em;'>
            AI-Powered Medical Image Analysis for Eye Disease Diagnosis
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Session state for result history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Patient details form
with st.form("patient_form"):
    st.subheader("👤 Details of the Patient")
    col1, col2 = st.columns(2)
    with col1:
        patient_name = st.text_input("Name of the Patient")
    with col2:
        patient_age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    st.markdown("#### 👁️ Upload Eye Images")
    left_eye_file = st.file_uploader("Upload Left Eye Image", type=["jpg", "jpeg", "png"], key="left")
    right_eye_file = st.file_uploader("Upload Right Eye Image", type=["jpg", "jpeg", "png"], key="right")
    submit = st.form_submit_button("Submit")

# Load model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('eye_disease_efficientnetb0_final.keras')
model = load_model()

# Class names (ensure order matches your training)
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def predict_eye(img_file):
    if img_file is None:
        return None, None
    img = Image.open(img_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions) * 100)
    return predicted_class, confidence

# On submit, make predictions and store in history
if submit:
    left_pred, left_acc = predict_eye(left_eye_file)
    right_pred, right_acc = predict_eye(right_eye_file)
    if None in (left_pred, right_pred):
        st.error("Please upload both left and right eye images.")
    else:
        result = {
            "Patient Name": patient_name,
            "Age": patient_age,
            "Gender": gender,
            "Left Eye Disease": left_pred,
            "Left Eye Accuracy (%)": f"{left_acc:.2f}",
            "Right Eye Disease": right_pred,
            "Right Eye Accuracy (%)": f"{right_acc:.2f}"
        }
        st.session_state['history'].append(result)
        st.success("Prediction complete! See results below.")

# Show results and allow CSV download
if st.session_state['history']:
    st.subheader("📋 Results History")
    df = pd.DataFrame(st.session_state['history'])
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="mediscan_results.csv",
        mime='text/csv'
    )
    if st.button("Delete History"):
        st.session_state['history'] = []
        st.success("History deleted.")

# Optional: add footer or instructions
st.markdown("---")
st.info("All predictions are AI-generated and for informational purposes only. Please consult a healthcare professional for medical advice.")
