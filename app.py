import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Load Dataset ---
# Assuming 'kc_house_data.csv.zip' is available in your Colab environment
data = pd.read_csv("kc_house_data.csv.zip")

# --- 2. Define Features (X) and Target (y) ---
# Using 'sqft_living', 'bedrooms', and 'bathrooms' as features
X = data[['sqft_living', 'bedrooms', 'bathrooms']]
y = data['price']  # The target variable is 'price'

# --- 3. Split Data into Training and Testing Sets ---
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 4. Initialize and Train the Model ---
# Using Linear Regression for prediction
model = LinearRegression()
model.fit(X_train, y_train)

# --- 5. Make Predictions ---
y_pred = model.predict(X_test)

# --- 6. Evaluate the Model ---
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"  Mean Absolute Error (MAE): {mae:.2f}")
print(f"  R-squared (R²): {r2:.2f}")

# --- 7. Example Prediction for a New House ---
# Create a DataFrame for a new house with specific features
# Ensure column names match the training features
new_house_data = pd.DataFrame({
    'sqft_living': [2000],  # Example: 2000 sqft living area
    'bedrooms': [4],        # Example: 4 bedrooms
    'bathrooms': [2.5]      # Example: 2.5 bathrooms
})

predicted_price = model.predict(new_house_data)

print("\nExample Prediction:")
print(f"  Predicted Price for a new house: ₹ {predicted_price[0]:.2f}")
