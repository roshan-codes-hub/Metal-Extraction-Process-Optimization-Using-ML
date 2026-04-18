import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('kiln_training_data.csv')
X = df[['Coal_Rate', 'RPM', 'Moisture', 'Feed_Temp']]
y = df['Nickel_Grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=5)
model.fit(X_train, y_train)

preds = model.predict(X_test)
r2  = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)

# Save everything needed by the app
with open('model_bundle.pkl', 'wb') as f:
    pickle.dump({ 'model': model, 'r2': r2, 'mae': mae }, f)

print(f"✅ Model saved — R²: {r2:.4f} | MAE: ±{mae:.2f}%")