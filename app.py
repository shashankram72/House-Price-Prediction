
import streamlit as st
import joblib

# Load trained model
model = joblib.load("house_price_model.pkl")

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    layout="centered"
)

# Heading
st.title("House Price Prediction")
st.write("Enter the house details to estimate its price.")
st.caption("Model: Linear Regression | Dataset: Ames Housing, USA")

st.divider()

# Input section
st.subheader("Enter House Details")

living_area = st.number_input(
    "Living Area (square feet)",
    min_value=300,
    max_value=10000,
    value=1500,
    step=100
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Number of Full Bathrooms",
    min_value=1,
    max_value=5,
    value=2
)

# Prediction button
if st.button("Predict House Price"):

    input_data = [[living_area, bedrooms, bathrooms]]

    predicted_price = model.predict(input_data)

    st.subheader("Estimated House Price")

    st.success(f"${predicted_price[0]:,.2f}")

    st.caption(
        "Estimated price based on the trained model. "
        "Actual market prices may differ."
    )
