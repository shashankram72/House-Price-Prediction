
import pandas as pd

# Load the housing dataset
data = pd.read_csv("dataset/train.csv")

# Display the first 5 rows
print("FIRST 5 ROWS OF THE DATASET:")
print(data.head())

# Display dataset size
print("\nDATASET SHAPE:")
print(data.shape)


# Display all column names
print("\nCOLUMN NAMES:")
print(data.columns.tolist())

# Display basic information
print("\nDATASET INFORMATION:")
data.info()

# Check missing values
print("\nMISSING VALUES:")
print(data.isnull().sum().sort_values(ascending=False).head(15))

# Understand house prices
print("\nHOUSE PRICE STATISTICS:")
print(data["SalePrice"].describe())


# Select input features
X = data[
    [
        "GrLivArea",
        "BedroomAbvGr",
        "FullBath",
        "OverallQual",
        "GarageCars",
        "TotalBsmtSF",
        "1stFlrSF",
        "YearBuilt",
        "TotRmsAbvGrd",
        "LotArea"
    ]
]

# Select target variable
y = data["SalePrice"]

# Display selected features
print("\nSELECTED FEATURES:")
print(X.head())

# Display target values
print("\nHOUSE PRICES:")
print(y.head())

# Check missing values in input features
print("\nMISSING VALUES IN X:")
print(X.isnull().sum())

# Check missing values in target
print("\nMISSING VALUES IN y:")
print(y.isnull().sum())


# Handle missing values in input features
X = X.copy()

X = X.fillna(X.median())

# Remove rows where target price is missing
valid_rows = y.notnull()

X = X.loc[valid_rows]
y = y.loc[valid_rows]

# Verify missing values are handled
print("\nMISSING VALUES AFTER CLEANING:")
print(X.isnull().sum())
print("Missing target values:", y.isnull().sum())

print("\nFINAL DATASET SIZE:")
print(X.shape)



from sklearn.model_selection import train_test_split

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Display the sizes
print("\nTRAINING DATA SIZE:")
print(X_train.shape)

print("\nTESTING DATA SIZE:")
print(X_test.shape)


from sklearn.linear_model import LinearRegression

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Calculate training and testing R2 scores
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("\nMODEL PERFORMANCE:")
print("Training R2 Score:", train_score)
print("Testing R2 Score:", test_score)

print("\nTRAINING SCORE (%):", train_score * 100)
print("TESTING SCORE (%):", test_score * 100)

print("\nMODEL TRAINED SUCCESSFULLY!")



# Predict house prices using testing data
y_pred = model.predict(X_test)

# Display first 10 predictions
print("\nPREDICTED HOUSE PRICES:")
print(y_pred[:10])

# Compare actual and predicted prices
comparison = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": y_pred[:10]
})

print("\nACTUAL VS PREDICTED PRICES:")
print(comparison)


from sklearn.metrics import mean_absolute_error

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

print("\nMODEL EVALUATION:")
print("Mean Absolute Error:", mae)


from sklearn.metrics import mean_squared_error

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

print("\nMEAN SQUARED ERROR:")
print(mse)


import matplotlib.pyplot as plt

# Create a scatter plot
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred, alpha=0.6)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

# Add an ideal prediction reference line
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    color="red",
    linestyle="--"
)

plt.show()

import joblib

# Save the trained model
joblib.dump(model, "house_price_model.pkl")

print("\nMODEL SAVED SUCCESSFULLY!")