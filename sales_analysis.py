import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Step 1: Create a synthetic dataset
np.random.seed(42)
data = {
    "TV_Ad_Spend": np.random.randint(10, 200, 50),
    "Radio_Ad_Spend": np.random.randint(5, 100, 50),
    "Online_Ad_Spend": np.random.randint(20, 300, 50),
    "Store_Size": np.random.randint(500, 2000, 50),
    "Holiday_Season": np.random.choice([0, 1], 50),  # 0 = No, 1 = Yes
}

df = pd.DataFrame(data)

# Sales formula (synthetic relationship)
df["Sales"] = (
    0.05 * df["TV_Ad_Spend"]
    + 0.04 * df["Radio_Ad_Spend"]
    + 0.03 * df["Online_Ad_Spend"]
    + 0.01 * df["Store_Size"]
    + 5 * df["Holiday_Season"]
    + np.random.normal(0, 3, 50)  # noise
)

print("Dataset sample:\n", df.head())

# Step 2: Features and target
X = df.drop("Sales", axis=1)
y = df["Sales"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("\nR² Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Step 7: Predict for a new store
new_data = pd.DataFrame({
    "TV_Ad_Spend": [120],
    "Radio_Ad_Spend": [50],
    "Online_Ad_Spend": [150],
    "Store_Size": [1200],
    "Holiday_Season": [1]
})

predicted_sales = model.predict(new_data)[0]
print("\nPredicted Sales for new store:", round(predicted_sales, 2))
