# ============================================
# Assignment 10
# Heart Disease Prediction using Machine Learning and Flask Deployment
# ============================================

# Import Libraries

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ============================================
# Task 1 : Data Understanding and Preprocessing
# ============================================

print("\n==============================")
print("Task 1 : Data Understanding and Preprocessing")
print("==============================")

df = pd.read_csv("heart.csv")

print("First Five Records:")
print(df.head())

numerical_features = df.drop("target", axis=1).columns.tolist()
target_variable = "target"

print("\nNumerical Features:")
print(numerical_features)

print("\nTarget Variable:")
print(target_variable)

print("\nMissing Values:")
print(df.isnull().sum())

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# ============================================
# Task 2 : Model Development
# ============================================

print("\n==============================")
print("Task 2 : Model Development")
print("==============================")

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy Score:")
print(round(accuracy * 100, 2), "%")

joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")

# ============================================
# Task 5 : Conclusion
# ============================================

print("\n==============================")
print("Task 5 : Conclusion")
print("==============================")

conclusion_text = """
Conclusion:

This assignment successfully developed a Heart Disease Prediction
system using the Random Forest Classification algorithm. The trained
model achieved high prediction accuracy and was saved using Joblib
for deployment through a Flask REST API. Deploying the model required
installing the necessary libraries, creating API endpoints, and
preparing the application for cloud hosting. This project demonstrates
the importance of MLOps by combining machine learning with deployment,
making the model accessible as a real-world prediction service. One
major advantage of this approach is that predictions can be delivered
quickly through an API, while a limitation is that deploying and
maintaining machine learning applications requires additional tools,
configuration, and cloud platform support.
"""

print(conclusion_text)
