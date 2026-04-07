# Machine Learning Algorithms + Feature Engineering Quick Guide

## 1. Feature Engineering

### Handle Missing Values
```python
df.fillna(df.mean(), inplace=True)
df.fillna(df.median(), inplace=True)
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)
````

### Remove Duplicates

```python
df.drop_duplicates(inplace=True)
```

### Encoding Categorical Data

Label Encoding

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["category"] = le.fit_transform(df["category"])
```

One Hot Encoding

```python
df = pd.get_dummies(df, columns=["category"])
```

### Feature Scaling

Standardization

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

Normalization

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
```

---

## Handle Outliers

### IQR Method

```python
Q1 = df["col"].quantile(0.25)
Q3 = df["col"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["col"] >= lower) & (df["col"] <= upper)]
```

### Z Score

```python
from scipy import stats
import numpy as np

z = np.abs(stats.zscore(df))
df = df[(z < 3).all(axis=1)]
```

### Clipping Outliers

```python
df["col"] = df["col"].clip(lower, upper)
```

---

## Feature Selection

Remove correlated features

```python
corr = df.corr()
```

Using feature importance

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X,y)

importance = model.feature_importances_
```

---

# 2. Algorithms (with Simple Code)

## Linear Regression (Regression)

Used when predicting **continuous values**

Example: house price

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)
```

---

## Logistic Regression (Classification)

Used for **binary classification**

Example: spam detection

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)
```

---

## KNN (K-Nearest Neighbors)

Works based on **distance between data points**

Good for small datasets

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
pred = model.predict(X_test)
```

---

## SVM (Support Vector Machine)

Good for **high dimensional data**

```python
from sklearn.svm import SVC

model = SVC(kernel="linear")
model.fit(X_train,y_train)
pred = model.predict(X_test)
```

---

## Decision Tree

Creates **tree-like decision structure**

Easy to interpret

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
pred = model.predict(X_test)
```

---

## Naive Bayes

Based on **Bayes theorem**

Works very well with **text data**

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train,y_train)
pred = model.predict(X_test)
```

---

## Ensemble Learning

Combines multiple models for better performance

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
pred = model.predict(X_test)
```

### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train,y_train)
pred = model.predict(X_test)
```

---

# 3. How To Choose Algorithm

| Data Situation           | Algorithm           |
| ------------------------ | ------------------- |
| Predict number           | Linear Regression   |
| Binary classification    | Logistic Regression |
| Small dataset            | KNN                 |
| High dimensional data    | SVM                 |
| Explainable model        | Decision Tree       |
| Text classification      | Naive Bayes         |
| Best general performance | Random Forest       |
| Very complex patterns    | Ensemble models     |

---

# 4. Quick ML Pipeline

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")

X = df.drop("target",axis=1)
y = df["target"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

pred = model.predict(X_test)
```

---

# 5. Memory Trick (For Exams)

```
Data → Clean → Feature Engineering → Split → Train → Predict → Evaluate
```

---

