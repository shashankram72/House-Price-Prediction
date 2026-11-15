# House Price Prediction Using Machine Learning

## Project Overview
This project predicts house prices using Machine Learning. It uses Linear Regression to estimate house prices based on selected house features.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib

## Dataset
The project uses the Ames Housing dataset from the Kaggle House Prices competition.

## Features Used
- Living Area (GrLivArea)
- Number of Bedrooms (BedroomAbvGr)
- Number of Full Bathrooms (FullBath)

## Machine Learning Model
Linear Regression

## Model Evaluation
- Mean Absolute Error (MAE): 35,788.06
- Mean Squared Error (MSE): 2,806,426,667.25

## Application
A Streamlit web application allows users to enter house details and receive an estimated house price.

## How to Run
1. Install the required libraries:
   `pip install pandas numpy matplotlib scikit-learn streamlit joblib`
2. Run the application:
   `streamlit run app.py`

## Project Structure
- dataset/train.csv — Housing dataset
- main.py — Model training and evaluation
- app.py — Web application
- house_price_model.pkl — Saved trained model

## Note
The predicted prices are estimates based on the training dataset and are not guaranteed market values.