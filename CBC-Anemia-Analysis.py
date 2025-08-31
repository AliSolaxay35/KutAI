# CBC Logistic Regression - Full Script for VS Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\AliSolaxay\OneDrive\Desktop\AITest\cbc_data_200.csv")
print("First 5 rows:")
print(df.head())

# General info
print("\nDataset Info:")
print(df.info())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# EDA - Visualization
plt.hist(df['Hemoglobin'], bins=20, edgecolor='black')
plt.title("Distribution of Hemoglobin")
plt.xlabel("Hemoglobin")
plt.ylabel("Count")
plt.show()

sns.boxplot(x="Anemia", y="Hemoglobin", data=df)
plt.title("Hemoglobin by Anemia Status")
plt.show()

sns.scatterplot(x="RBC", y="Hemoglobin", hue="Anemia", data=df)
plt.title("RBC vs Hemoglobin")
plt.show()

# Prepare features and target
X = df[["Hemoglobin", "RBC", "WBC", "Platelets"]]
y = df["Anemia"]

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
