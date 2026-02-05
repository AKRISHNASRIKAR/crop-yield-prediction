import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🌾 Crop Yield & Smart Irrigation",
    layout="wide"
)

st.title("🌾 Crop Yield Prediction & Smart Irrigation System")
st.caption("Machine Learning–based Decision Support for Sustainable Agriculture")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("sample_crop_data.csv")

df = load_data()

# =====================================================
# PREPROCESSING
# =====================================================
le_crop = LabelEncoder()
le_season = LabelEncoder()

df["Crop_Type"] = le_crop.fit_transform(df["Crop_Type"])
df["Season"] = le_season.fit_transform(df["Season"])

features = [
    "Crop_Type", "Season", "Temperature", "Rainfall", "Humidity",
    "Soil_Moisture", "Soil_pH", "Nitrogen", "Phosphorus", "Potassium"
]

X = df[features]
y = df["Crop_Yield"]

# =====================================================
# TRAIN MODEL
# =====================================================
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        random_state=42
    )
    model.fit(X_train, y_train)

    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2

model, r2 = train_model(X, y)

# =====================================================
# SIDEBAR INPUT
# =====================================================
st.sidebar.header("🌱 Input Parameters")

crop = st.sidebar.selectbox(
    "Crop Type", le_crop.classes_
)
season = st.sidebar.selectbox(
    "Season", le_season.classes_
)

temp = st.sidebar.slider("Temperature (°C)", 15.0, 40.0, 28.0)
rain = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
humidity = st.sidebar.slider("Humidity (%)", 40.0, 90.0, 65.0)
soil_m = st.sidebar.slider("Soil Moisture (%)", 20.0, 80.0, 45.0)
ph = st.sidebar.slider("Soil pH", 5.0, 8.0, 6.5)
n = st.sidebar.slider("Nitrogen (kg/ha)", 20.0, 100.0, 50.0)
p = st.sidebar.slider("Phosphorus (kg/ha)", 15.0, 80.0, 40.0)
k = st.sidebar.slider("Potassium (kg/ha)", 20.0, 90.0, 45.0)

# =====================================================
# PREDICTION
# =====================================================
input_df = pd.DataFrame([[
    le_crop.transform([crop])[0],
    le_season.transform([season])[0],
    temp, rain, humidity, soil_m, ph, n, p, k
]], columns=features)

predicted_yield = model.predict(input_df)[0]

# =====================================================
# IRRIGATION LOGIC
# =====================================================
def irrigation_advice(rainfall, soil_moisture):
    if rainfall > 50 or soil_moisture > 60:
        return 5
    elif soil_moisture > 40:
        return 15
    else:
        return 25

recommended_water = irrigation_advice(rain, soil_m)

# =====================================================
# MAIN OUTPUT
# =====================================================
col1, col2, col3 = st.columns(3)

col1.metric("📊 Predicted Yield", f"{predicted_yield:.2f} tons/ha")
col2.metric("💧 Recommended Irrigation", f"{recommended_water} mm/week")
col3.metric("📈 Model Accuracy (R²)", f"{r2:.2f}")

st.divider()

# =====================================================
# WHAT-IF SIMULATOR 
# =====================================================
st.header("🧪 What-If Scenario Simulator")
st.caption("Simulate improvements and observe impact on yield")

delta_n = st.slider("Increase Nitrogen (kg/ha)", 0, 30, 10)
delta_rain = st.slider("Increase Rainfall (mm)", 0, 50, 20)

what_if_df = input_df.copy()
what_if_df["Nitrogen"] += delta_n
what_if_df["Rainfall"] += delta_rain

new_yield = model.predict(what_if_df)[0]

fig, ax = plt.subplots()
ax.bar(
    ["Original", "What-If"],
    [predicted_yield, new_yield],
    color=["#6c757d", "#2ecc71"]
)
ax.set_ylabel("Yield (tons/ha)")
ax.set_title("Yield Comparison")
st.pyplot(fig)

st.success(
    f"📈 Yield Improvement: **{(new_yield - predicted_yield):.2f} tons/ha**"
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
st.divider()
st.header("🔍 Feature Importance")

importance = model.feature_importances_
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(features, importance)
ax2.set_title("Feature Contribution to Yield Prediction")
st.pyplot(fig2)
