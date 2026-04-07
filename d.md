Great, let’s break this down step by step so you can implement both tasks clearly in Python. I’ll outline the workflow, show you the code, and explain the reasoning behind each part.

---

## 1. Classification with KNN (mobile.csv)

We’ll use the **K-Nearest Neighbors (KNN)** algorithm to classify whether a customer will purchase a mobile based on **Age** and **Estimated Salary**.

### Steps:
1. Load the dataset (`mobile.csv`).
2. Visualize the distribution of purchased vs not purchased customers.
3. Apply **MinMaxScaler** to normalize features.
4. Split into train/test sets.
5. Train KNN classifier.
6. Predict and evaluate.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load dataset
data = pd.read_csv("mobile.csv")

# Step 2: Visualization
plt.scatter(data['Age'][data['Purchased']==0], data['EstimatedSalary'][data['Purchased']==0], color='red', label='Not Purchased')
plt.scatter(data['Age'][data['Purchased']==1], data['EstimatedSalary'][data['Purchased']==1], color='green', label='Purchased')
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

# Step 3: Feature scaling
X = data[['Age','EstimatedSalary']]
y = data['Purchased']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Step 6: Prediction
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

👉 This will give you a scatter plot showing the distribution, and the KNN model will predict purchase behavior.

---

## 2. Regression with computer.csv

Here we’ll determine which regression model fits best for the dataset. Typically, **Linear Regression** is suitable if the target variable is continuous.

### Steps:
1. Load dataset (`computer.csv`).
2. Check rows and columns.
3. Choose regression model (Linear Regression).
4. Train model, fetch intercept & coefficients.
5. Visualize regression line.
6. Compute **Coefficient of Determination (R²)**.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Load dataset
data = pd.read_csv("computer.csv")

# Step 2: Shape of dataset
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

# Assume 'Price' is target and others are features
X = data.drop('Price', axis=1)
y = data['Price']

# Step 3: Regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Intercept & Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Step 5: Visualization (example with one feature, say 'Speed')
plt.scatter(data['Speed'], y, color='blue')
plt.plot(data['Speed'], model.predict(X), color='red')
plt.xlabel("Speed")
plt.ylabel("Price")
plt.title("Linear Regression Fit")
plt.show()

# Step 6: Accuracy (R²)
y_pred = model.predict(X)
print("Coefficient of Determination (R²):", r2_score(y, y_pred))
```
