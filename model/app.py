import streamlit as st
import pandas as pd
import xgboost as xgb
from scipy.optimize import dual_annealing

# Page Config
st.set_page_config(page_title="AI Kiln Control Optimizer", layout="wide")
st.title("🏭 Smart Rotary Kiln Control System")
st.markdown("### Prescriptive AI for Nickel Laterite Reduction")

# 1. Load and Cache the AI Model
@st.cache_resource
def load_model():
    df = pd.read_csv('kiln_training_data.csv')
    X = df[['Coal_Rate', 'RPM', 'Moisture', 'Feed_Temp']]
    y = df['Nickel_Grade']
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X, y)
    return model

ai_model = load_model()

# 2. Sidebar for Environmental Disturbances (Things you can't control)
st.sidebar.header("📡 Real-time Sensor Inputs")
moist = st.sidebar.slider("Ore Moisture (%)", 10.0, 35.0, 20.0)
ftemp = st.sidebar.slider("Feed Temperature (°C)", 150.0, 300.0, 250.0)
target = st.sidebar.number_input("Target Nickel Grade (%)", value=85.0)

# 3. Optimization Logic
if st.button("🚀 Find Optimal Setpoints"):
    
    def objective(setpoints):
        coal, rpm = setpoints
        pred = ai_model.predict(pd.DataFrame([[coal, rpm, moist, ftemp]], 
                                            columns=['Coal_Rate', 'RPM', 'Moisture', 'Feed_Temp']))[0]
        return (pred - target)**2

    # Global search for setpoints
    res = dual_annealing(objective, bounds=[(1000, 2000), (0.5, 2.0)], maxiter=50)
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