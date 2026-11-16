
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
    value=1500
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

quality = st.number_input(
    "Overall Quality (1-10)",
    min_value=1,
    max_value=10,
    value=5
)

garage_cars = st.number_input(
    "Garage Capacity",
    min_value=0,
    max_value=5,
    value=2
)

basement_area = st.number_input(
    "Basement Area (sq ft)",
    min_value=0,
    max_value=5000,
    value=800
)

first_floor = st.number_input(
    "First Floor Area (sq ft)",
    min_value=300,
    max_value=5000,
    value=1000
)

year_built = st.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2000
)

total_rooms = st.number_input(
    "Total Rooms",
    min_value=2,
    max_value=15,
    value=6
)

lot_area = st.number_input(
    "Lot Area (sq ft)",
    min_value=1000,
    max_value=100000,
    value=8000
)

# Prediction button
if st.button("Predict House Price"):

    input_data = [[
        living_area,
        bedrooms,
        bathrooms,
        quality,
        garage_cars,
        basement_area,
        first_floor,
        year_built,
        total_rooms,
        lot_area
    ]]

    predicted_price = model.predict(input_data)

    st.subheader("Estimated House Price")

    st.success(f"${predicted_price[0]:,.2f}")

    st.caption(
        "Estimated price based on the trained model. "
        "Actual market prices may differ."
    )
