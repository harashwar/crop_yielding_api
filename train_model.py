"""
╔══════════════════════════════════════════════════════════════════════╗
║   AI-Powered Crop Yield Prediction and Optimization                  ║
║   train_model.py — Model Training & Evaluation Script               ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

# For plotting learning curves
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# ──────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("  AI-Powered Crop Yield Prediction")
print("  Model Training Pipeline")
print("=" * 60)

CSV_PATH = "soil_dataset_2000_with_yield.csv"
df = pd.read_csv(CSV_PATH)

print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head().to_string(index=False))

# ──────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("📊 DATASET STATISTICS")
print("─" * 60)
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget variable statistics:")
print(df['Yield_tons_per_hectare'].describe())

print(f"\nUnique values per categorical column:")
for col in ['district', 'crop', 'Soil_Type']:
    print(f"  {col}: {sorted(df[col].unique())}")

# ──────────────────────────────────────────────────────────
# 3. PREPROCESSING — Label Encoding
# ──────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔧 PREPROCESSING")
print("─" * 60)

categorical_cols = ['district', 'crop', 'Soil_Type']
label_encoders = {}

df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Features and Target
FEATURE_COLS = ['district', 'N', 'P', 'K', 'OC', 'pH', 'EC', 'S', 'Fe', 'Zn', 'Cu', 'B', 'Mn', 'temperature', 'humidity', 'crop', 'Soil_Type']
TARGET_COL   = 'Yield_tons_per_hectare'

X = df_encoded[FEATURE_COLS]
y = df_encoded[TARGET_COL]

# ──────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n✅ Split: Train={len(X_train)} rows, Test={len(X_test)} rows")

# ──────────────────────────────────────────────────────────
# 5. TRAIN MODELS
# ──────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🌲 TRAINING MODELS")
print("─" * 60)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# ──────────────────────────────────────────────────────────
# 5b. LEARNING CURVES (TRAIN/VAL)
# ──────────────────────────────────────────────────────────
def plot_learning_curve(model, X, y, model_name):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='r2', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True, random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_mean, 'o-', label='Training R²')
    plt.plot(train_sizes, val_mean, 'o-', label='Validation R²')
    plt.title(f'Learning Curve: {model_name}')
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────
# 6. EVALUATE MODELS
# ──────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("📈 MODEL EVALUATION")
print("─" * 60)

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n  ┌─ {name} ─────────────────────────")
    print(f"  │  MAE  (Mean Absolute Error) : {mae:.4f} Ton/Hectare")
    print(f"  │  RMSE (Root Mean Sq Error)  : {rmse:.4f} Ton/Hectare")
    print(f"  │  R²   (Accuracy Score)      : {r2:.4f} ({r2*100:.1f}%)")
    print(f"  └──────────────────────────────────")
    return mae, rmse, r2

rf_mae, rf_rmse, rf_r2 = evaluate("Random Forest Regressor", y_test, rf_pred)
gb_mae, gb_rmse, gb_r2 = evaluate("Gradient Boosting Regressor", y_test, gb_pred)

# Cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print(f"\n  Cross-Validation (5-Fold) R² Scores: {cv_scores.round(4)}")
print(f"  Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Choose best model
best_model = rf_model if rf_r2 >= gb_r2 else gb_model
best_name  = "Random Forest" if rf_r2 >= gb_r2 else "Gradient Boosting"
print(f"\n  🏆 Best Model: {best_name} (R² = {max(rf_r2, gb_r2):.4f})")

# Plot learning curve for the best model
print(f"\n📉 Plotting learning curve for {best_name}...")
plot_learning_curve(best_model, X, y, best_name)

# ──────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("📌 FEATURE IMPORTANCE (Random Forest)")
print("─" * 60)

importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=False)
for feat, imp in importances.items():
    bar = "█" * int(imp * 40)
    print(f"  {feat:<15} {imp:.4f}  {bar}")

# ──────────────────────────────────────────────────────────
# 8. SAVE MODEL & METADATA
# ──────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("💾 SAVING MODEL & ARTIFACTS")
print("─" * 60)

os.makedirs("model", exist_ok=True)

# Save best model
joblib.dump(best_model, "model/crop_yield_model.pkl")
print("  ✅ Saved: model/crop_yield_model.pkl")

# Save label encoders
joblib.dump(label_encoders, "model/label_encoders.pkl")
print("  ✅ Saved: model/label_encoders.pkl")

# Save metadata as JSON
metadata = {
    "model_name"       : best_name,
    "features"         : FEATURE_COLS,
    "target"           : TARGET_COL,
    "train_size"       : len(X_train),
    "test_size"        : len(X_test),
    "rf_mae"           : round(rf_mae, 4),
    "rf_rmse"          : round(rf_rmse, 4),
    "rf_r2"            : round(rf_r2, 4),
    "gb_mae"           : round(gb_mae, 4),
    "gb_rmse"          : round(gb_rmse, 4),
    "gb_r2"            : round(gb_r2, 4),
    "cv_mean_r2"       : round(float(cv_scores.mean()), 4),
    "cv_std_r2"        : round(float(cv_scores.std()), 4),
    "districts"        : sorted(df['district'].unique().tolist()),
    "crop_types"       : sorted(df['crop'].unique().tolist()),
    "soil_types"       : sorted(df['Soil_Type'].unique().tolist()),
    "humidity_range"   : [int(df['humidity'].min()), int(df['humidity'].max())],
    "temperature_range": [int(df['temperature'].min()), int(df['temperature'].max())],
}
with open("model/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("  ✅ Saved: model/model_metadata.json")

# ──────────────────────────────────────────────────────────
# 9. SAMPLE PREDICTIONS
# ──────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("🔮 SAMPLE PREDICTIONS ON NEW DATA")
print("─" * 60)

sample_inputs = [
    {"district": "Thanjavur", "N": 50, "P": 40, "K": 30, "OC": 50, "pH": 7.0, "EC": 20, "S": 10, "Fe": 10, "Zn": 5, "Cu": 2, "B": 2, "Mn": 10, "temperature": 30, "humidity": 70, "crop": "Rice", "Soil_Type": "Loamy"}
]

print(f"\n  Testing sample prediction...")
for row in sample_inputs:
    row_enc = row.copy()
    row_enc['district']  = label_encoders['district'].transform([row['district']])[0]
    row_enc['crop']      = label_encoders['crop'].transform([row['crop']])[0]
    row_enc['Soil_Type'] = label_encoders['Soil_Type'].transform([row['Soil_Type']])[0]
    
    input_df = pd.DataFrame([row_enc], columns=FEATURE_COLS)
    pred = best_model.predict(input_df)[0]
    print(f"  {row['district']} - {row['Soil_Type']} - {row['crop']}: Predicted Yield = {pred:>5.3f} T/Hectare")

print("\n" + "=" * 60)
print("✅ Training complete! Run `python app.py` to start the web app.")
print("=" * 60)
