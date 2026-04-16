import streamlit as st
import pandas as pd
import xgboost as xgb
from scipy.optimize import dual_annealing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ══════════════════════════════════════════════════════
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="Ni-Kiln AI Optimizer", layout="wide", page_icon="🏭")

# Custom CSS for a modern "Dark Industrial" look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4255; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    h1 { color: #ffffff; text-shadow: 2px 2px 4px #000000; }
    </style>
    """,  unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# 2. DATA & MODEL LOADING (WITH ERROR HANDLING)
# ══════════════════════════════════════════════════════
@st.cache_resource
def load_and_train_model():
    try:
        # Update filename to match your repo exactly
        df = pd.read_csv('kiln_training_data.csv') 
        X = df[['Coal_Rate', 'RPM', 'Moisture', 'Feed_Temp']]
        y = df['Nickel_Grade']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=5)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        return model, r2, mae, None
    except Exception as e:
        return None, 0, 0, str(e)

ai_model, r2_val, mae_val, error_msg = load_and_train_model()

# ══════════════════════════════════════════════════════
# 3. HEADER SECTION
# ══════════════════════════════════════════════════════
st.title("🏭 Rotary Kiln: Prescriptive AI Controller")
st.markdown("---")

if error_msg:
    st.error(f"⚠️ Error loading data: {error_msg}")
    st.stop()

# Accuracy Row
col_acc1, col_acc2, col_acc3 = st.columns(3)
with col_acc1:
    st.metric("Model Precision (R²)", f"{r2_val:.4f}")
with col_acc2:
    st.metric("Avg Error (MAE)", f"±{mae_val:.2f}% Ni")
with col_acc3:
    st.metric("Status", "Operational" if r2_val > 0.9 else "Calibrating")

st.markdown("###") # Spacer

# ══════════════════════════════════════════════════════
# 4. CONTROL ROOM (INPUTS & SIMULATION)
# ══════════════════════════════════════════════════════
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    with st.container():
        st.subheader("📡 Live Sensor Data")
        st.info("Adjust the disturbances below to simulate current kiln conditions.")
        
        moist = st.slider("💧 Ore Moisture (%)", 5.0, 45.0, 18.0)
        ftemp = st.slider("🔥 Feed Temperature (°C)", 150.0, 600.0, 250.0)
        target = st.number_input("🎯 Target Nickel Grade (%)", value=85.0, step=0.5)
        
        st.write("---")
        run_opt = st.button("🚀 OPTIMIZE SET-POINTS")

with right_col:
    st.subheader("⚙️ Optimized Set-points")
    
    if run_opt:
        # Optimization Logic
        def objective(u):
            coal, rpm = u
            input_df = pd.DataFrame([[coal, rpm, moist, ftemp]], 
                                   columns=['Coal_Rate', 'RPM', 'Moisture', 'Feed_Temp'])
            pred = ai_model.predict(input_df)[0]
            # Quadratic Error + Efficiency Penalty (Small coal bias)
            return (pred - target)**2 + (0.001 * coal)

        with st.spinner("Analyzing Thermodynamics..."):
            res = dual_annealing(objective, bounds=[(1000, 2500), (0.5, 2.5)], maxiter=40)
            opt_coal, opt_rpm = res.x
            
            # Final Prediction for confirmation
            final_pred = ai_model.predict(pd.DataFrame([[opt_coal, opt_rpm, moist, ftemp]], 
                                         columns=['Coal_Rate', 'RPM', 'Moisture', 'Feed_Temp']))[0]

        # Results Cards
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.success(f"**Coal Feed Rate**\n### {opt_coal:.1f} kg/h")
        with res_col2:
            st.success(f"**Kiln Speed**\n### {opt_rpm:.2f} RPM")
        with res_col3:
            if abs(final_pred - target) < 1.0:
                st.success(f"**Predicted Nickel Grade**\n### {final_pred:.2f}%")
            else:
                st.warning(f"**Predicted Nickel Grade**\n### {final_pred:.2f}%")
        
        st.write("---")
        if abs(final_pred - target) >= 1.0:
            st.warning("Note: The AI suggests this is the best possible grade achievable under current high-moisture conditions.")
    else:
        st.info("Awaiting sensor data... Click 'Optimize Set-points' to generate the control plan.")


# ══════════════════════════════════════════════════════
# 6. ANALYTICS & INSIGHTS SECTION
# ══════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📊 Analytics & Insights")

tab1, tab2 = st.tabs(["🧬 Genetic Algorithm Optimization", "🔬 Feature Correlation Analysis"])

with tab1:
    if st.button("▶ Run GA Analysis"):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        with st.spinner("Running Genetic Algorithm..."):
            # ── GA Parameters ──
            POP_SIZE, N_GEN, MUTATION_RATE = 60, 50, 0.15
            COAL_MIN, COAL_MAX = 1000, 2500
            RPM_MIN, RPM_MAX  = 0.5, 2.5

            def predict_grade(coal, rpm):
                inp = pd.DataFrame([[coal, rpm, moist, ftemp]],
                                   columns=['Coal_Rate','RPM','Moisture','Feed_Temp'])
                return ai_model.predict(inp)[0]

            # Initialise population
            pop = np.column_stack([
                np.random.uniform(COAL_MIN, COAL_MAX, POP_SIZE),
                np.random.uniform(RPM_MIN,  RPM_MAX,  POP_SIZE)
            ])

            best_per_gen, avg_per_gen = [], []
            best_ind, best_grade = pop[0], -np.inf

            for gen in range(N_GEN):
                fitness = np.array([predict_grade(c, r) for c, r in pop])
                best_idx = np.argmax(fitness)
                gen_best = fitness[best_idx]
                best_per_gen.append(gen_best)
                avg_per_gen.append(fitness.mean())

                if gen_best > best_grade:
                    best_grade = gen_best
                    best_ind   = pop[best_idx].copy()

                # Selection (top 50%)
                sorted_idx = np.argsort(fitness)[::-1]
                parents    = pop[sorted_idx[:POP_SIZE // 2]]

                # Crossover
                children = []
                for _ in range(POP_SIZE // 2):
                    p1, p2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
                    child  = (p1 + p2) / 2
                    children.append(child)
                children = np.array(children)

                # Mutation
                mask = np.random.rand(*children.shape) < MUTATION_RATE
                noise = np.random.randn(*children.shape) * np.array([50, 0.1])
                children[mask] += noise[mask]
                children[:, 0] = np.clip(children[:, 0], COAL_MIN, COAL_MAX)
                children[:, 1] = np.clip(children[:, 1], RPM_MIN,  RPM_MAX)

                pop = np.vstack([parents, children])

            opt_coal_ga, opt_rpm_ga = best_ind
            opt_grade_ga = best_grade

        # ── Plot ──
        fig = plt.figure(figsize=(16, 5), facecolor='#0e1117')
        fig.suptitle(f"Genetic Algorithm Optimization  →  Optimal: Coal={opt_coal_ga:.0f} kg/h, "
                     f"RPM={opt_rpm_ga:.2f}, Grade={opt_grade_ga:.1f}%",
                     color='white', fontsize=12)
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        ax_style = dict(facecolor='#1e2130', labelcolor='white',
                        titlecolor='white', grid=True)

        def style_ax(ax, title, xlabel, ylabel):
            ax.set_facecolor('#1e2130')
            ax.set_title(title, color='white', fontsize=10)
            ax.set_xlabel(xlabel, color='white', fontsize=9)
            ax.set_ylabel(ylabel, color='white', fontsize=9)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#3e4255')
            ax.grid(True, color='#3e4255', linestyle='--', alpha=0.5)

        # Plot 1 – Convergence
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(best_per_gen, color='royalblue',  lw=2, label='Best grade')
        ax1.plot(avg_per_gen,  color='salmon',     lw=1.5, linestyle='--', label='Avg grade')
        ax1.axhline(opt_grade_ga, color='limegreen', linestyle=':', lw=1.5, label=f'Optimal: {opt_grade_ga:.1f}%')
        ax1.legend(fontsize=8, facecolor='#1e2130', labelcolor='white')
        style_ax(ax1, "GA Convergence — Best Grade per Generation", "Generation", "Predicted Nickel Grade (%)")

        # Plot 2 – Fitness Landscape heatmap
        ax2 = fig.add_subplot(gs[1])
        coal_grid = np.linspace(COAL_MIN, COAL_MAX, 40)
        rpm_grid  = np.linspace(RPM_MIN,  RPM_MAX,  40)
        Z = np.array([[predict_grade(c, r) for c in coal_grid] for r in rpm_grid])
        im = ax2.contourf(coal_grid, rpm_grid, Z, levels=20, cmap='RdYlGn')
        fig.colorbar(im, ax=ax2, label='Predicted Nickel Grade (%)').ax.yaxis.label.set_color('white')
        ax2.scatter(opt_coal_ga, opt_rpm_ga, marker='*', s=200, color='royalblue', zorder=5, label='GA Optimum')
        ax2.legend(fontsize=8, facecolor='#1e2130', labelcolor='white')
        style_ax(ax2, "Fitness Landscape\n(Grade across all Coal + RPM settings)",
                 "Coal Feed Rate (kg/h)", "Kiln RPM (r/min)")

        # Plot 3 – Sensitivity Analysis
        ax3 = fig.add_subplot(gs[2])
        deviations = np.linspace(-30, 30, 60)
        sens_coal = [predict_grade(opt_coal_ga * (1 + d/100), opt_rpm_ga)  for d in deviations]
        sens_rpm  = [predict_grade(opt_coal_ga, opt_rpm_ga  * (1 + d/100)) for d in deviations]
        ax3.plot(deviations, sens_coal, color='royalblue',  lw=2,      label='Coal Rate ±%')
        ax3.plot(deviations, sens_rpm,  color='salmon',     lw=2, linestyle='--', label='Kiln RPM ±%')
        ax3.axhline(opt_grade_ga, color='limegreen', linestyle=':', lw=1.5, label=f'Optimum ({opt_grade_ga:.1f}%)')
        ax3.axhline(target,       color='orange',    linestyle='--', lw=1.5, label=f'Target ({target:.0f}%)')
        ax3.legend(fontsize=8, facecolor='#1e2130', labelcolor='white')
        style_ax(ax3, "Sensitivity Analysis\n(Grade impact of operator deviation)",
                 "Deviation from Optimal Setting (%)", "Predicted Nickel Grade (%)")

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Click **▶ Run GA Analysis** to run the Genetic Algorithm and visualize convergence, fitness landscape, and sensitivity.")

with tab2:
    if st.button("▶ Show Feature Correlations"):
        import numpy as np
        import matplotlib.pyplot as plt

        try:
            df_plot = pd.read_csv('kiln_training_data.csv')
            features = ['Coal_Rate', 'RPM', 'Moisture', 'Feed_Temp']

            fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor='#0e1117')
            fig.suptitle("Feature Correlation with Nickel Grade", color='white', fontsize=12)

            for ax, feat in zip(axes, features):
                corr = df_plot[feat].corr(df_plot['Nickel_Grade'])
                ax.scatter(df_plot[feat], df_plot['Nickel_Grade'],
                           alpha=0.5, color='steelblue', edgecolors='none', s=25)
                ax.set_facecolor('#1e2130')
                ax.set_title(f"{feat}\ncorr={corr:.3f}", color='white', fontsize=10)
                ax.set_xlabel(feat, color='white', fontsize=9)
                ax.set_ylabel("Nickel Grade (%)", color='white', fontsize=9)
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#3e4255')
                ax.grid(True, color='#3e4255', linestyle='--', alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Could not load data for plotting: {e}")
    else:
        st.info("Click **▶ Show Feature Correlations** to view scatter plots of each input variable vs. Nickel Grade.")

# ══════════════════════════════════════════════════════
# 5. FOOTER / PROJECT INFO
# ══════════════════════════════════════════════════════
st.markdown("---")
st.caption("Developed for Materials Processing Optimization | Hybrid Mechanistic-XGBoost Architecture")
