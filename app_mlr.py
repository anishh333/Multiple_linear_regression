import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as slt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# page config
slt.set_page_config("Multiple Linear Regression", layout='centered')

# css
def load_css(file):
    with open(file) as f:
        slt.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

slt.markdown("""
    <div class="card">
        <h1> Multiple Linear Regression</h1>
        <p> Predict the tip amount based on size and total bill</p>
    </div>
""", unsafe_allow_html=True)

# Load Data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# Preview
slt.subheader("DataSet Preview")
slt.dataframe(df[["total_bill", "size", "tip"]].head())

# preparing data
x, y = df[["total_bill", "size"]], df["tip"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Standardization
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# model
model = LinearRegression()
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

# metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

p = x.shape[1]  # number of features
n = len(y_test)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# plotting (VALID visualization)
slt.subheader("Tip vs Total Bill (Colored by Size)")

fig, ax = plt.subplots()
scatter = ax.scatter(
    df["total_bill"],
    df["tip"],
    c=df["size"],
    cmap="viridis",
    alpha=0.7
)
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
plt.colorbar(scatter, label="Size")

slt.pyplot(fig)

# Metrics - Performance
c1, c2 = slt.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("MSE", f"{mse:.2f}")

c3, c4 = slt.columns(2)
c3.metric("R2", f"{r2:.2f}")
c4.metric("Adj R2", f"{adj_r2:.2f}")

# Interpretation
slt.markdown(f"""
    <div class="card">
        <h3>Model Interpretation</h3>
        <p><b>Total Bill Coefficient:</b> {model.coef_[0]:.3f}</p>
        <p><b>Size Coefficient:</b> {model.coef_[1]:.3f}</p>
        <p><b>Intercept:</b> {model.intercept_:.3f}</p>
    </div>
""", unsafe_allow_html=True)

# Predictions
bill = slt.slider(
    "Enter the Bill amount",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

size = slt.slider(
    "Enter no of people",
    int(df['size'].min()),
    int(df['size'].max()),
    2
)

pred_tip = model.predict(scaler.transform([[bill, size]]))[0]

slt.markdown(
    f'<div class="prediction-box"> Predicted tip is $ {pred_tip:.2f}</div>',
    unsafe_allow_html=True
)
