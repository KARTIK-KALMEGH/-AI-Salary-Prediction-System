import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("../dataset/salary_data.csv")

print("Dataset loaded successfully")
print("Shape:", data.shape)

# Encode categorical columns
le_dict = {}
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Features and target
X = data[[
    "Total_Experience",
    "Current_CTC",
    "No_Of_Companies_worked",
    "Last_Appraisal_Rating"
]]

y = data["Expected_CTC"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Linear regression
lr = LinearRegression()
lr.fit(X_train,y_train)
pred1 = lr.predict(X_test)

print("\nLinear Regression R2:", r2_score(y_test,pred1))

# Random forest
rf = RandomForestRegressor(n_estimators=200,random_state=42)
rf.fit(X_train,y_train)
pred2 = rf.predict(X_test)

print("Random Forest R2:", r2_score(y_test,pred2))
print("MAE:", mean_absolute_error(y_test,pred2))

# Save model
joblib.dump(rf,"../model/salary_model.pkl")

print("\nMODEL TRAINED & SAVED SUCCESSFULLY 🔥")