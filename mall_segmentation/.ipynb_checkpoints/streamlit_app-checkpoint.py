from pathlib import Path

import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from joblib import load

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent / "data"
ENCODER_PATH = DATA_DIR / "encoder_model.keras"
KMEANS_PATH  = DATA_DIR / "autoencoder_kmeans.joblib"
SCALER_PATH  = DATA_DIR / "scaler.joblib"

# Gender is dropped — model uses only these 3 features
FEATURES = ["Age", "Annual Income", "Spending Score"]

CLUSTER_NAMES = {
    0: "Budget-Conscious Customers",
    1: "VIP / High Spenders",
    2: "Average Customers",
    3: "Impulsive Buyers",
}

CLUSTER_DESC = {
    0: "Low income, low spending. Price-sensitive shoppers who respond well to discounts and deals.",
    1: "High income, high spending. Premium customers — worth targeting with exclusive offers and loyalty programs.",
    2: "Middle income, moderate spending. The typical mall visitor, broad appeal.",
    3: "Mixed income but very high spending. Impulse-driven — respond well to flash sales and promotions.",
}


# ── Load models ───────────────────────────────────────────────────────────────
def load_models():
    encoder = tf.keras.models.load_model(ENCODER_PATH)
    kmeans  = load(KMEANS_PATH)
    scaler  = load(SCALER_PATH)
    return encoder, kmeans, scaler


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_cluster(age, annual_income, spending_score, encoder, kmeans, scaler):
    new_customer = np.array(
        [[age, annual_income, spending_score]],
        dtype=np.float32,
    )
    new_customer_scaled = scaler.transform(new_customer)
    latent  = encoder.predict(new_customer_scaled, verbose=0)
    cluster = int(kmeans.predict(latent)[0])
    return latent[0], cluster


class MallSegmentApp:
    def __init__(self, root, encoder, kmeans, scaler):
        self.root = root
        self.encoder = encoder
        self.kmeans = kmeans
        self.scaler = scaler

        self.root.title("Mall Customer Segment Predictor")
        self.root.geometry("760x520")
        self.root.minsize(700, 480)

        self.age_var = tk.IntVar(value=30)
        self.income_var = tk.DoubleVar(value=60000.0)
        self.score_var = tk.DoubleVar(value=50.0)

        self.cluster_var = tk.StringVar(value="Predicted Segment: -")
        self.desc_var = tk.StringVar(value="")
        self.latent_var = tk.StringVar(value="-")

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        title = ttk.Label(
            main,
            text="Mall Customer Segment Predictor",
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            main,
            text="Autoencoder (3 -> 2 latent dims) + K-Means clustering",
        )
        subtitle.pack(anchor="w", pady=(2, 14))

        input_frame = ttk.LabelFrame(main, text="Enter Customer Details", padding=12)
        input_frame.pack(fill="x")

        ttk.Label(input_frame, text="Age").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Spinbox(input_frame, from_=12, to=100, textvariable=self.age_var, width=12).grid(
            row=0, column=1, sticky="w", pady=6
        )

        ttk.Label(input_frame, text="Annual Income (RM)").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=6
        )
        ttk.Spinbox(
            input_frame,
            from_=0,
            to=500000,
            increment=1000,
            textvariable=self.income_var,
            width=12,
            format="%.0f",
        ).grid(row=1, column=1, sticky="w", pady=6)

        ttk.Label(input_frame, text="Spending Score (1-100)").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=6
        )
        ttk.Spinbox(
            input_frame,
            from_=1,
            to=100,
            increment=1,
            textvariable=self.score_var,
            width=12,
            format="%.0f",
        ).grid(row=2, column=1, sticky="w", pady=6)

        ttk.Button(input_frame, text="Predict Segment", command=self.on_predict).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0)
        )

        result_frame = ttk.LabelFrame(main, text="Prediction Result", padding=12)
        result_frame.pack(fill="both", expand=True, pady=(14, 0))

        ttk.Label(result_frame, textvariable=self.cluster_var, font=("Segoe UI", 12, "bold")).pack(anchor="w")

        self.desc_label = ttk.Label(result_frame, textvariable=self.desc_var, wraplength=680, justify="left")
        self.desc_label.pack(anchor="w", pady=(8, 10))

        ttk.Label(result_frame, text="Latent vector (compressed representation):").pack(anchor="w")
        latent_box = ttk.Entry(result_frame, textvariable=self.latent_var, state="readonly")
        latent_box.pack(fill="x", pady=(4, 10))

        notes = (
            "How it works:\n"
            "1. Inputs are standardized using the saved scaler.\n"
            "2. The encoder compresses 3 features into 2 latent dimensions.\n"
            "3. K-Means assigns the nearest cluster in latent space.\n"
            "Gender is excluded from the model."
        )
        ttk.Label(result_frame, text=notes, justify="left").pack(anchor="w")

    def on_predict(self):
        try:
            age = int(self.age_var.get())
            annual_income = float(self.income_var.get())
            spending_score = float(self.score_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid input", "Please enter valid numeric values.")
            return

        if not 12 <= age <= 100:
            messagebox.showerror("Invalid age", "Age must be between 12 and 100.")
            return
        if not 0 <= annual_income <= 500000:
            messagebox.showerror("Invalid income", "Annual income must be between 0 and 500000.")
            return
        if not 1 <= spending_score <= 100:
            messagebox.showerror("Invalid score", "Spending score must be between 1 and 100.")
            return

        latent_vector, predicted_cluster = predict_cluster(
            age=age,
            annual_income=annual_income,
            spending_score=spending_score,
            encoder=self.encoder,
            kmeans=self.kmeans,
            scaler=self.scaler,
        )

        cluster_name = CLUSTER_NAMES.get(predicted_cluster, f"Segment {predicted_cluster}")
        cluster_desc = CLUSTER_DESC.get(predicted_cluster, "")

        self.cluster_var.set(f"Predicted Segment: {cluster_name}")
        self.desc_var.set(cluster_desc)
        self.latent_var.set(np.array2string(latent_vector.round(4), precision=4))


def main():
    try:
        encoder_model, kmeans_model, scaler_model = load_models()
    except Exception as exc:
        messagebox.showerror("Startup error", f"Failed to load model artifacts:\n{exc}")
        return

    root = tk.Tk()
    app = MallSegmentApp(root, encoder_model, kmeans_model, scaler_model)
    root.mainloop()


if __name__ == "__main__":
    main()
