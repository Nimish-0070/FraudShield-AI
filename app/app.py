import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "fraud_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    features = joblib.load(MODEL_DIR / "feature_names.pkl")
    return model, scaler, features

try:
    model, scaler, feature_names = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# Use your validation-selected threshold here, if available.
THRESHOLD = 0.50

# -----------------------------
# PREPROCESSING
# -----------------------------
def prepare_data(data):
    data = data[feature_names].copy()

    # Scale only the two features used during training
    data[["Time", "Amount"]] = scaler.transform(
        data[["Time", "Amount"]]
    )

    return data

def predict_transactions(data):
    processed = prepare_data(data)
    probabilities = model.predict_proba(processed)[:, 1]

    results = data.copy()
    results["Fraud Probability (%)"] = probabilities * 100
    results["Prediction"] = np.where(
        probabilities >= THRESHOLD,
        "Fraudulent",
        "Legitimate"
    )

    return results

# -----------------------------
# HEADER
# -----------------------------
st.title("🛡️ FraudShield AI")
st.subheader("Intelligent Credit Card Fraud Detection & Risk Analysis")
st.caption("Machine Learning powered transaction risk assessment")

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Single Transaction",
        "Bulk CSV Prediction",
        "About Model"
    ]
)

st.sidebar.divider()
st.sidebar.caption("FraudShield AI | Portfolio Project")

# -----------------------------
# DASHBOARD
# -----------------------------
if page == "Dashboard":

    st.header("Fraud Detection Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Status", "Loaded")
    col2.metric("Input Features", len(feature_names))
    col3.metric("Fraud Threshold", f"{THRESHOLD:.2f}")

    st.info(
        "Upload transaction data in the Bulk CSV Prediction section "
        "or test an individual transaction."
    )

    st.markdown("### Available Features")

    st.write("""
    - Individual transaction risk prediction
    - Bulk transaction analysis
    - Fraud probability scoring
    - Model-based classification
    """)

# -----------------------------
# SINGLE TRANSACTION ANALYSIS
# -----------------------------
elif page == "Single Transaction":

    st.header("Single Transaction Analysis")

    st.write(
        "Select a sample transaction from the benchmark dataset "
        "and analyze its fraud risk."
    )

    # Dataset location
    DATA_PATH = BASE_DIR.parent / "data" / "creditcard.csv"

    @st.cache_data
    def load_sample_data():
        return pd.read_csv(DATA_PATH)

    try:
        sample_data = load_sample_data()

        # Select sample transactions
        sample_indices = sample_data.sample(
            n=min(20, len(sample_data)),
            random_state=42
        ).index.tolist()

        selected_index = st.selectbox(
            "Choose a Sample Transaction",
            sample_indices,
            format_func=lambda x: f"Sample Transaction #{x}"
        )

        # Get selected transaction
        selected_row = sample_data.loc[selected_index]

        # Keep only model input features
        transaction = selected_row[feature_names]

        transaction_df = pd.DataFrame(
            [transaction],
            columns=feature_names
        )

        # Display transaction details
        st.subheader("Transaction Information")

        col1, col2 = st.columns(2)

        col1.metric(
            "Transaction Amount",
            f"${selected_row['Amount']:.2f}"
        )

        col2.metric(
            "Transaction Time",
            f"{selected_row['Time']:.2f} seconds"
        )

        st.info(
            "The 28 anonymized PCA features are loaded "
            "automatically from the selected dataset record."
        )

        # Analyze transaction
        if st.button(
            "Analyze Transaction",
            type="primary"
        ):

            prediction = predict_transactions(transaction_df)

            probability = prediction[
                "Fraud Probability (%)"
            ].iloc[0]

            result = prediction[
                "Prediction"
            ].iloc[0]

            st.divider()
            st.subheader("Fraud Detection Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Fraud Probability",
                f"{probability:.2f}%"
            )

            col2.metric(
                "Classification",
                result
            )

            if result == "Fraudulent":
                st.error(
                    "Potential Fraud Detected"
                )
            else:
                st.success(
                    "Transaction Classified as Legitimate"
                )

            # Show actual dataset label for comparison
            if "Class" in sample_data.columns:

                actual_label = int(selected_row["Class"])

                actual_result = (
                    "Fraudulent"
                    if actual_label == 1
                    else "Legitimate"
                )

                st.subheader("Benchmark Dataset Comparison")

                st.write(
                    f"Actual dataset label: **{actual_result}**"
                )

                if result == actual_result:
                    st.success(
                        "The model prediction matches the dataset label."
                    )
                else:
                    st.warning(
                        "The model prediction differs from the dataset label."
                    )

            st.caption(
                "This is a demonstration using a historical "
                "benchmark dataset, not a live banking transaction."
            )

    except FileNotFoundError:
        st.error(
            "Dataset not found. Please place creditcard.csv "
            "inside the project's data folder."
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
# -----------------------------
# BULK CSV PREDICTION
# -----------------------------
elif page == "Bulk CSV Prediction":

    st.header("Bulk Transaction Analysis")

    st.write(
        "Upload a CSV containing the 30 model input features: "
        "Time, V1–V28 and Amount."
    )

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        missing_columns = [
            col for col in feature_names
            if col not in data.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required columns: {missing_columns}"
            )
        else:
            if st.button("Run Fraud Detection", type="primary"):

                try:
                    results = predict_transactions(data)

                    st.success("Predictions generated successfully!")

                    total = len(results)
                    fraud_count = (
                        results["Prediction"] == "Fraudulent"
                    ).sum()
                    legitimate_count = total - fraud_count

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Total Transactions", total)
                    col2.metric("Fraudulent", int(fraud_count))
                    col3.metric("Legitimate", int(legitimate_count))

                    st.subheader("Prediction Results")
                    st.dataframe(results, use_container_width=True)

                    csv_data = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download Predictions CSV",
                        data=csv_data,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# -----------------------------
# ABOUT MODEL
# -----------------------------
elif page == "About Model":

    st.header("About FraudShield AI")

    st.markdown("""
    **FraudShield AI** is a machine learning project designed to
    identify potentially fraudulent credit card transactions.

    ### Dataset
    - European credit card transaction dataset
    - 284,807 original records
    - 30 input features
    - Binary classification: legitimate or fraudulent

    ### Machine Learning
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - XGBoost

    ### Explainability
    - SHAP global feature importance
    - SHAP summary analysis

    ### Important limitation

    This application is a research and portfolio demonstration.
    Its predictions should not be used as the sole basis for
    real-world financial decisions.
    """)