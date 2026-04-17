from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from joblib import load
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).parent / "data"
ENCODER_PATH = DATA_DIR / "encoder_model.keras"
KMEANS_PATH = DATA_DIR / "autoencoder_kmeans.joblib"
RAW_DATA_PATH = DATA_DIR / "Shopping_Mall_Customer_Segmentation_Data_.csv"

FEATURES = ["Age", "Gender", "Annual Income", "Spending Score"]
NUMERIC_FEATURES = ["Age", "Annual Income", "Spending Score"]

CLUSTER_NAMES = {
    0: "Budget-Conscious Customers",
    1: "VIP / High Spenders",
    2: "Average Customers",
    3: "Impulsive Buyers",
}


@st.cache_resource
def load_models():
    encoder = tf.keras.models.load_model(ENCODER_PATH)
    kmeans = load(KMEANS_PATH)
    return encoder, kmeans


@st.cache_data
def load_dataset_and_scaler():
    df = pd.read_csv(RAW_DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    required_columns = [
        "Age",
        "Gender",
        "Annual Income",
        "Spending Score",
    ]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    model_df = df[required_columns].copy()
    model_df["Gender"] = model_df["Gender"].astype(str).str.strip().str.capitalize()
    model_df["Gender"] = model_df["Gender"].map({"Female": 0, "Male": 1}).fillna(1).astype(int)

    scaler = StandardScaler()
    scaler.fit(model_df[FEATURES])
    return df, scaler


def predict_cluster(age, gender, annual_income, spending_score, encoder, kmeans, scaler):
    gender_encoded = 0 if gender == "Female" else 1
    new_customer = np.array(
        [[age, gender_encoded, annual_income, spending_score]],
        dtype=np.float32,
    )
    new_customer_scaled = scaler.transform(new_customer)
    latent = encoder.predict(new_customer_scaled, verbose=0)
    cluster = int(kmeans.predict(latent)[0])
    return latent[0], cluster


st.set_page_config(page_title="Mall Customer Segment Predictor", page_icon="🛍️", layout="centered")
st.title("Mall Customer Segment Predictor")
st.caption("Autoencoder + K-Means clustering based segment prediction")

try:
    df_raw, scaler = load_dataset_and_scaler()
    encoder_loaded, kmeans_loaded = load_models()
except Exception as exc:
    st.error(f"Failed to load app artifacts: {exc}")
    st.stop()

with st.form("predict_form"):
    age = st.number_input("Age", min_value=12, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", ["Female", "Male"])
    annual_income = st.number_input("Annual Income", min_value=0.0, max_value=200000.0, value=60000.0, step=1000.0)
    spending_score = st.number_input("Spending Score (1-100)", min_value=1.0, max_value=100.0, value=50.0, step=1.0)
    submitted = st.form_submit_button("Predict Segment")

if submitted:
    latent_vector, predicted_cluster = predict_cluster(
        age=age,
        gender=gender,
        annual_income=annual_income,
        spending_score=spending_score,
        encoder=encoder_loaded,
        kmeans=kmeans_loaded,
        scaler=scaler,
    )

    result_df = df_raw.copy()
    result_df.columns = [c.strip() for c in result_df.columns]
    result_df["Gender"] = result_df["Gender"].astype(str).str.strip().str.capitalize()
    result_df["Gender"] = result_df["Gender"].map({"Female": 0, "Male": 1}).fillna(1).astype(int)

    scaled_all = scaler.transform(result_df[FEATURES])
    latent_all = encoder_loaded.predict(scaled_all, verbose=0)
    result_df["Cluster"] = kmeans_loaded.predict(latent_all)

    cluster_avg = result_df[result_df["Cluster"] == predicted_cluster][NUMERIC_FEATURES].mean()

    st.subheader("Prediction Result")
    st.success(f"Predicted Segment: {CLUSTER_NAMES.get(predicted_cluster, f'Segment {predicted_cluster}')}")

    col1, col2 = st.columns(2)
    col1.metric("Cluster ID", predicted_cluster)
    col2.metric("Latent Vector", np.array2string(latent_vector.round(4), precision=4))

    st.markdown("### Cluster Profile (Average)")
    st.write(f"Age: {cluster_avg['Age']:.0f}")
    st.write(f"Annual Income: {cluster_avg['Annual Income']:,.0f}")
    st.write(f"Spending Score: {cluster_avg['Spending Score']:.0f}")

    st.info(
        "How this works:\n"
        "1. Input features are standardized using training-distribution statistics.\n"
        "2. The encoder compresses inputs into latent representation.\n"
        "3. K-Means assigns the nearest latent cluster."
    )