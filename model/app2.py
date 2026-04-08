import streamlit as st
import pandas as pd
import xgboost as xgb
from scipy.optimize import dual_annealing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Page Config
st.set_page_config(page_title="AI Kiln Control Optimizer", layout="wide")
st.title("🏭 Smart Rotary Kiln Control System")
st.markdown("### Prescriptive AI for Nickel Laterite Reduction")

# 1. Load and Cache the AI Model
@st.cache_resource
def load_model():
    df = pd.read_csv('kiln_training_data2.csv')
    X = df[['Coal_Rate', 'RPM', 'Moisture', 'Feed_Temp']]
    y = df['Nickel_Grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Calculate Metrics
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    return model, r2, mae

ai_model, r2_val, mae_val = load_model()

# Add a "Model Reliability" section to the UI
# Unpack the results from the loader

# Create three columns for the header metrics
m1, m2, m3 = st.columns(3)

with m1:
    st.metric(label="AI Prediction Accuracy (R²)", value=f"{r2_val:.4f}")
    st.caption("Closer to 1.0 is better")

with m2:
    st.metric(label="Mean Absolute Error (MAE)", value=f"±{mae_val:.2f}%")
    st.caption("Average deviation in Nickel Grade")

with m3:
    status = "High Reliability" if r2_val > 0.90 else "Low Reliability"
    st.metric(label="Model Status", value=status)
    st.caption("Based on 80/20 Train-Test Split")

st.divider()


# 2. Sidebar for Environmental Disturbances (Things you can't control)
st.sidebar.header("📡 Real-time Sensor Inputs")
moist = st.sidebar.slider("Ore Moisture (%)", 5.0, 45.0, 20.0)
ftemp = st.sidebar.slider("Feed Temperature (°C)", 100.0, 500.0, 250.0)
target = st.sidebar.number_input("Target Nickel Grade (%)", value=85.0)

# 3. Optimization Logic
if st.button("🚀 Find Optimal Setpoints"):
    
    def objective(setpoints):
        coal, rpm = setpoints
        pred = ai_model.predict(pd.DataFrame([[coal, rpm, moist, ftemp]], 
                                            columns=['Coal_Rate', 'RPM', 'Moisture', 'Feed_Temp']))[0]
        return (pred - target)**2

    # Global search for setpoints
    res = dual_annealing(objective, bounds=[(800, 2500), (0.3, 2.5)], maxiter=50)
    opt_coal, opt_rpm = res.x
    final_grade = ai_model.predict(pd.DataFrame([[opt_coal, opt_rpm, moist, ftemp]], 
                                               columns=['Coal_Rate', 'RPM', 'Moisture', 'Feed_Temp']))[0]

    # 4. Display Results
    col1, col2, col3 = st.columns(3)
    col1.metric("Recommended Coal Rate", f"{opt_coal:.1f} kg/h")
    col2.metric("Recommended Kiln Speed", f"{opt_rpm:.2f} RPM")
    col3.metric("Predicted Resulting Grade", f"{final_grade:.2f}%")
    
    if abs(final_grade - target) < 1.0:
        st.success("Target Achievable with these settings!")
    else:
        st.warning("Target difficult to reach with current moisture levels.")