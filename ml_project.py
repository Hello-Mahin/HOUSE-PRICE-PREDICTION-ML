
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create dataset
data = {
    "Area": [1000, 1500, 2000, 2500, 3000, 3500],
    "Bedrooms": [2, 3, 3, 4, 4, 5],
    "Price": [200000, 300000, 400000, 500000, 600000, 700000]
}

df = pd.DataFrame(data)

# Features & Target
X = df[["Area", "Bedrooms"]]
y = df["Price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score:.2f}")

# Prediction
area = int(input("Enter Area: "))
bedrooms = int(input("Enter Bedrooms: "))

prediction = model.predict([[area, bedrooms]])
print(f"Predicted Price: {prediction[0]}")