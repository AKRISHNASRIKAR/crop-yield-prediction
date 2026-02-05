import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Crop Yield Prediction & Smart Irrigation",
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
# PREPROCESSING (FIXED: PER CROP)
# =====================================================

# Sidebar crop selection FIRST
st.sidebar.header("🌱 Input Parameters")
crop = st.sidebar.selectbox(
    "Crop Type",
    sorted(df["Crop_Type"].unique())
)

# Filter dataset by selected crop
df_crop = df[df["Crop_Type"] == crop].copy()

# Safety check
if len(df_crop) < 30:
    st.error(f"Not enough data for {crop} (only {len(df_crop)} samples) to train a reliable model.")
    st.stop()

# One-hot encode ONLY season
df_encoded = pd.get_dummies(
    df_crop,
    columns=["Season"],
    drop_first=True
)

X = df_encoded.drop(["Crop_Yield", "Crop_Type"], axis=1)
y = df_encoded["Crop_Yield"]

# =====================================================
# TRAIN MODEL
# =====================================================
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))

    return model, r2, X.columns

model, r2, feature_columns = train_model(X, y)

# =====================================================
# SIDEBAR INPUTS
# =====================================================
# Crop selected above

season = st.sidebar.selectbox("Season", sorted(df["Season"].unique()))

temperature = st.sidebar.slider("Temperature (°C)", 15.0, 40.0, 28.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
humidity = st.sidebar.slider("Humidity (%)", 40.0, 90.0, 65.0)
soil_moisture = st.sidebar.slider("Soil Moisture (%)", 20.0, 80.0, 45.0)
soil_ph = st.sidebar.slider("Soil pH", 5.0, 8.0, 6.5)

nitrogen = st.sidebar.slider("Nitrogen (kg/ha)", 20.0, 120.0, 50.0)
phosphorus = st.sidebar.slider("Phosphorus (kg/ha)", 15.0, 100.0, 40.0)
potassium = st.sidebar.slider("Potassium (kg/ha)", 20.0, 110.0, 45.0)

historical_irrigation = st.sidebar.slider(
    "Historical Irrigation Water (mm)",
    0.0, 300.0, 100.0,
    help="Average irrigation water used in previous seasons"
)

# =====================================================
# BUILD INPUT VECTOR
# =====================================================
def build_input_row(
    temp, rain, hum, sm, ph, n, p, k, hist_irrigation, season
):
    row = {
        "Temperature": temp,
        "Rainfall": rain,
        "Humidity": hum,
        "Soil_Moisture": sm,
        "Soil_pH": ph,
        "Nitrogen": n,
        "Phosphorus": p,
        "Potassium": k,
        "Historical_Irrigation_Water": hist_irrigation
    }

    for col in feature_columns:
        if col.startswith("Season_"):
            row[col] = 1 if col == f"Season_{season}" else 0

    # Create DataFrame and ensure column order matches training data
    df_row = pd.DataFrame([row])
    # Reindex checks if all columns are present and strictly orders them
    df_row = df_row.reindex(columns=feature_columns, fill_value=0)
    
    return df_row

# =====================================================
# CURRENT PREDICTION
# =====================================================
input_df = build_input_row(
    temperature, rainfall, humidity,
    soil_moisture, soil_ph,
    nitrogen, phosphorus, potassium,
    historical_irrigation,
    season
)

predicted_yield = model.predict(input_df)[0]

# =====================================================
# IRRIGATION RECOMMENDATION LOGIC
# =====================================================
def irrigation_recommendation(rain, soil_m):
    if rain > 50 or soil_m > 60:
        return 5
    elif soil_m > 40:
        return 15
    else:
        return 25

recommended_water = irrigation_recommendation(rainfall, soil_moisture)

# =====================================================
# MAIN METRICS
# =====================================================
col1, col2, col3 = st.columns(3)

col1.metric("🌾 Predicted Yield", f"{predicted_yield:.2f} tons/ha")
col2.metric("💧 Recommended Irrigation", f"{recommended_water} mm/week")
col3.metric("📊 Model Accuracy (R²)", f"{r2:.2f}")

st.divider()

# =====================================================
# 🔮 FUTURE YIELD PREDICTION
# =====================================================
st.header("🔮 Future Yield Prediction")
st.caption("Predict crop yield under expected future conditions")

col_f1, col_f2 = st.columns(2)

with col_f1:
    f_temp = st.slider("Expected Temperature (°C)", 15.0, 40.0, temperature)
    f_rain = st.slider("Expected Rainfall (mm)", 0.0, 300.0, rainfall)
    f_humidity = st.slider("Expected Humidity (%)", 40.0, 90.0, humidity)
    f_soil_m = st.slider("Expected Soil Moisture (%)", 20.0, 80.0, soil_moisture)

with col_f2:
    f_ph = st.slider("Expected Soil pH", 5.0, 8.0, soil_ph)
    f_n = st.slider("Planned Nitrogen (kg/ha)", 20.0, 120.0, nitrogen)
    f_p = st.slider("Planned Phosphorus (kg/ha)", 15.0, 100.0, phosphorus)
    f_k = st.slider("Planned Potassium (kg/ha)", 20.0, 110.0, potassium)

future_df = build_input_row(
    f_temp, f_rain, f_humidity,
    f_soil_m, f_ph,
    f_n, f_p, f_k,
    historical_irrigation,
    season
)

future_yield = model.predict(future_df)[0]
yield_diff = future_yield - predicted_yield

col_a, col_b, col_c = st.columns(3)

col_a.metric("Current Yield", f"{predicted_yield:.2f}")
col_b.metric("Future Yield", f"{future_yield:.2f}")
col_c.metric("Change", f"{yield_diff:+.2f}")

if yield_diff > 0.3:
    st.success("✅ Future conditions are favorable. Yield is expected to improve.")
elif yield_diff > 0:
    st.info("ℹ️ Slight yield improvement expected.")
elif yield_diff > -0.3:
    st.warning("⚠️ Slight yield reduction possible. Monitor conditions.")
else:
    st.error("🚨 Significant yield reduction expected. Reassess planning.")

# =====================================================
# COMPARISON CHART (COMPACT)
# =====================================================
fig, ax = plt.subplots(figsize=(4.5, 3))
ax.bar(
    ["Current", "Future"],
    [predicted_yield, future_yield],
    color=["#3498db", "#2ecc71"]
)
ax.set_ylabel("Yield (tons/ha)")
ax.set_title("Yield Forecast Comparison")
st.pyplot(fig, use_container_width=False)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
st.divider()
st.header("🔍 Feature Importance")

fi_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.barh(fi_df["Feature"], fi_df["Importance"])
ax2.set_title("Feature Contribution to Yield Prediction")
ax2.tick_params(labelsize=8)
st.pyplot(fig2)
