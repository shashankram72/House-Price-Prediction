# House Price Prediction Using Machine Learning

## Project Overview
This project predicts house prices using Machine Learning. It uses Linear Regression to estimate house prices based on different house features.

The project is developed using Python and Streamlit with the Ames Housing dataset.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib

## Dataset
Dataset: Ames Housing Dataset (Kaggle)

- Total records: 1460
- Target variable: SalePrice
- Location represented by dataset: Ames, Iowa, USA
- Currency: USD

## Features Used
The model uses the following 10 features:

1. GrLivArea – Above-ground living area
2. BedroomAbvGr – Number of bedrooms
3. FullBath – Number of full bathrooms
4. OverallQual – Overall house quality
5. GarageCars – Garage capacity
6. TotalBsmtSF – Total basement area
7. 1stFlrSF – First-floor area
8. YearBuilt – Year of construction
9. TotRmsAbvGrd – Total rooms above ground
10. LotArea – Total land area

## Machine Learning Algorithm
**Linear Regression**

Linear Regression is used to identify the relationship between house features and their selling prices.

## Model Evaluation

The model was initially developed using 3 features and later improved using 10 features.

| Evaluation Metric | Original Model | Improved Model |
|---|---:|---:|
| Training R² | 56.52% | 77% |
| Testing R² | 63.41% | 80% |
| MAE | $35,788.06 | $24,434.80 |
| MSE | 2,806,426,667.25 | 1,501,294,637.24 |

The improved model achieved an 80% testing R² score.

## Model Saving Using Joblib

Joblib is used to save and load the trained machine learning model.

- `joblib.dump()` saves the trained model.
- `joblib.load()` loads the saved model.

The trained model is stored in `house_price_model.pkl` and reused in the Streamlit application.

## Project Structure

```text
House_Price_Prediction/
│
├── dataset/
│   └── train.csv
│
├── main.py
├── app.py
├── house_price_model.pkl
├── readme.md
├── requirements.txt
└── .gitignore
```

## How to Run the Project

### 1. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python main.py
```

This trains the Linear Regression model and saves it as `house_price_model.pkl`.

### 3. Run the Web Application

```bash
streamlit run app.py
```

The application opens in your web browser.

## Application Features
- Enter 10 different house features.
- Predict estimated house prices.
- Display the predicted price.
- Simple and user-friendly Streamlit interface.

## Conclusion

This project demonstrates how Machine Learning can be used to predict house prices using Linear Regression. By increasing the number of input features from 3 to 10, the testing R² score improved from 63.41% to 80%, and the Mean Absolute Error decreased.

## Disclaimer

This model is trained on the Ames Housing dataset from the USA. Its predictions are estimates based on that dataset and should not be treated as actual market prices in other locations.
