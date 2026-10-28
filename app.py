from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="House Price Predictor | EduBridge Mini Project 1",
    page_icon="🏠",
    layout="centered",
)

MODEL_PATH = Path(__file__).resolve().parent / "house_price_model.pkl"


@st.cache_resource
def load_trained_model():
    """Load the trained model bundle from disk."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        import pickle

        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)


model_bundle = load_trained_model()

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: #f0f2f6;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #31333f;
        margin-bottom: 0.5rem;
    }
    .prediction-card {
        padding: 1.5rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .prediction-title {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }
    .prediction-value {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        font-size: 0.85rem;
        color: #888888;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    '<div class="main-header">'
    '<span class="badge">EduBridge • Mini Project 1</span>'
    "<h1>🏠 House Price Predictor</h1>"
    "<p style='color: #555;'>Estimate property market value using a trained Machine Learning model based on key real estate features.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Verify model presence
if model_bundle is None:
    st.error(
        "❌ **Model file (`house_price_model.pkl`) not found!**\n\n"
        "Please run `python train_model.py` in your terminal to train and generate the model file first."
    )
    st.stop()

model = model_bundle["model"]
features = model_bundle.get("features", ["area_sqft", "bedrooms", "bathrooms", "age_years"])
model_name = model_bundle.get("model_name", "Linear Regression")
metrics = model_bundle.get("metrics", {})

# User Input Form
with st.form("prediction_form"):
    st.subheader("📋 Property Specifications")

    col1, col2 = st.columns(2)

    with col1:
        area_sqft = st.number_input(
            "Square Footage (sq ft)",
            min_value=300,
            max_value=10_000,
            value=1_500,
            step=50,
            help="Total living area in square feet.",
        )
        bedrooms = st.slider(
            "Number of Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Total bedrooms in the house.",
        )

    with col2:
        bathrooms = st.slider(
            "Number of Bathrooms",
            min_value=1,
            max_value=8,
            value=2,
            step=1,
            help="Total bathrooms.",
        )
        age_years = st.number_input(
            "Property Age (years)",
            min_value=0,
            max_value=100,
            value=8,
            step=1,
            help="Age of the house in years.",
        )

    submitted = st.form_submit_button("🔍 Calculate Estimated Price", use_container_width=True)

# Prediction Logic
if submitted:
    input_data = pd.DataFrame(
        [[area_sqft, bedrooms, bathrooms, age_years]],
        columns=features,
    )
    prediction = model.predict(input_data)[0]
    estimated_price = max(0, prediction)
    price_per_sqft = estimated_price / area_sqft if area_sqft > 0 else 0

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-title">Estimated Market Value</div>
            <div class="prediction-value">${estimated_price:,.0f}</div>
            <div>Avg: ~${price_per_sqft:,.2f} per sq ft</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Area", f"{area_sqft:,} sqft")
    col_m2.metric("Bedrooms", f"{bedrooms}")
    col_m3.metric("Bathrooms", f"{bathrooms}")
    col_m4.metric("Age", f"{age_years} yrs")

# Expandable Model Information
with st.expander("ℹ️ About the Machine Learning Model"):
    st.write(f"**Selected Algorithm:** `{model_name}`")
    st.write(f"**Input Features:** `{', '.join(features)}`")
    if metrics:
        st.write(f"**Test R² Score:** `{metrics.get('r2_score', 'N/A')}`")
        st.write(f"**Mean Absolute Error (MAE):** `${metrics.get('mae', 0):,.2f}`")
        st.write(f"**Root Mean Squared Error (RMSE):** `${metrics.get('rmse', 0):,.2f}`")
    st.caption("Developed as part of EduBridge Mini Project 1 curriculum.")

# Footer
st.markdown(
    '<div class="footer">'
    "House Price Prediction • EduBridge Mini Project 1 • Built with Streamlit & Scikit-Learn"
    "</div>",
    unsafe_allow_html=True,
)
