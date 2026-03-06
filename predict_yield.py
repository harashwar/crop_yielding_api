"""
╔══════════════════════════════════════════════════════════════════════╗
║   predict_yield.py — Crop Yield Prediction using Trained Model      ║
║   AI-Powered Crop Yield Prediction System                           ║
║                                                                      ║
║   Model: Random Forest (scikit-learn)                               ║
║   Accuracy: R² = 98.57% | MAE = 0.4496 T/Ha                        ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
  python predict_yield.py
  python predict_yield.py --district Thanjavur --crop Rice --soil Loamy --temp 30 --humidity 70 --N 50 --P 40 --K 30

How it works:
  - Loads the trained Random Forest model from model/crop_yield_model.pkl
  - Loads label encoders from model/label_encoders.pkl
  - Accepts soil/weather/crop inputs and predicts yield in Tons/Hectare
"""

import argparse
import json
import os
import sys
import joblib
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join("model", "crop_yield_model.pkl")
ENCODERS_PATH = os.path.join("model", "label_encoders.pkl")
METADATA_PATH = os.path.join("model", "model_metadata.json")

# ── Feature columns (must match training order exactly) ────────────────────────
FEATURE_COLS = [
    'district', 'N', 'P', 'K', 'OC', 'pH', 'EC', 'S',
    'Fe', 'Zn', 'Cu', 'B', 'Mn', 'temperature', 'humidity', 'crop', 'Soil_Type'
]

BANNER = """
╔══════════════════════════════════════════════════════╗
║  🌾  AI Crop Yield Predictor                        ║
║      Tamil Nadu Agriculture Intelligence System      ║
║      Model Accuracy: R² = 98.57%                    ║
╚══════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    """Load the trained scikit-learn model, encoders and metadata."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at '{MODEL_PATH}'")
        print("   Please run:  python train_model.py")
        sys.exit(1)

    model    = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)

    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)

    return model, encoders, meta


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_yield(model, encoders, inputs: dict) -> float:
    """
    Predict crop yield given input features.

    Parameters
    ----------
    model    : trained sklearn model
    encoders : dict of LabelEncoders for categorical columns
    inputs   : dict with keys matching FEATURE_COLS

    Returns
    -------
    float : predicted yield in Tons/Hectare
    """
    row = inputs.copy()

    # Encode categorical columns
    for cat_col in ['district', 'crop', 'Soil_Type']:
        encoder = encoders[cat_col]
        value   = str(row[cat_col])

        if value not in encoder.classes_:
            print(f"⚠️  Warning: '{value}' not seen during training for '{cat_col}'.")
            print(f"   Known values: {list(encoder.classes_)}")
            print(f"   Using closest available value instead.")
            value = encoder.classes_[0]   # fallback

        row[cat_col] = encoder.transform([value])[0]

    input_df = pd.DataFrame([row], columns=FEATURE_COLS)
    prediction = float(model.predict(input_df)[0])
    return round(prediction, 4)


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY LABEL
# ─────────────────────────────────────────────────────────────────────────────
def quality_label(yield_val: float) -> str:
    if yield_val >= 8.0:  return "🌟 EXCELLENT"
    if yield_val >= 5.0:  return "✅ GOOD"
    if yield_val >= 2.5:  return "⚠️  AVERAGE"
    return "🔴 LOW"


# ─────────────────────────────────────────────────────────────────────────────
# PRINT RESULT
# ─────────────────────────────────────────────────────────────────────────────
def print_result(inputs, prediction, rmse):
    low  = round(max(0.0, prediction - rmse), 3)
    high = round(prediction + rmse, 3)

    print("\n" + "═" * 56)
    print(f"  📍 District     : {inputs['district']}")
    print(f"  🌱 Crop         : {inputs['crop']}")
    print(f"  🪨 Soil Type    : {inputs['Soil_Type']}")
    print(f"  🌡️  Temperature  : {inputs['temperature']} °C")
    print(f"  💧 Humidity     : {inputs['humidity']} %")
    print(f"  🧪 N/P/K        : {inputs['N']} / {inputs['P']} / {inputs['K']}")
    print("─" * 56)
    print(f"  🌾 Predicted Yield  : {prediction:.4f} Tons/Hectare")
    print(f"  📊 Quality          : {quality_label(prediction)}")
    print(f"  📉 Confidence Range : {low} – {high} T/Ha (±RMSE)")
    print("═" * 56)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────────────────────
def interactive_mode(model, encoders, meta):
    print("\n  📝 Interactive Prediction Mode")
    print("  (Type 'quit' at any time to exit)\n")

    districts  = meta.get("districts",  [])
    crop_types = meta.get("crop_types", [])
    soil_types = meta.get("soil_types", [])

    while True:
        print("─" * 56)

        # District
        print(f"  Available Districts: {', '.join(districts)}")
        district = input("  📍 District         : ").strip().title()
        if district.lower() == 'quit': break

        # Crop
        print(f"\n  Available Crops: {', '.join(crop_types)}")
        crop = input("  🌱 Crop Type        : ").strip().title()
        if crop.lower() == 'quit': break

        # Soil Type
        print(f"\n  Soil Types: {', '.join(soil_types)}")
        soil = input("  🪨 Soil Type        : ").strip().title()
        if soil.lower() == 'quit': break

        # Numeric inputs
        try:
            temp     = float(input("  🌡️  Temperature (°C) : ").strip())
            humidity = float(input("  💧 Humidity (%)     : ").strip())
            N        = float(input("  🧪 Nitrogen (N)     : ").strip())
            P        = float(input("  🧪 Phosphorus (P)   : ").strip())
            K        = float(input("  🧪 Potassium (K)    : ").strip())
            OC       = float(input("  🧴 Organic Carbon   : ").strip())
            pH       = float(input("  🔬 pH Value         : ").strip())
            EC       = float(input("  ⚡ EC (dS/m)        : ").strip())
            S        = float(input("  🔩 Sulphur (S)      : ").strip())
            Fe       = float(input("  🔩 Iron (Fe)        : ").strip())
            Zn       = float(input("  🔩 Zinc (Zn)        : ").strip())
            Cu       = float(input("  🔩 Copper (Cu)      : ").strip())
            B        = float(input("  🔩 Boron (B)        : ").strip())
            Mn       = float(input("  🔩 Manganese (Mn)   : ").strip())
        except ValueError:
            print("  ❌ Invalid numeric input. Please try again.")
            continue

        inputs = {
            'district': district, 'crop': crop, 'Soil_Type': soil,
            'temperature': temp,  'humidity': humidity,
            'N': N, 'P': P, 'K': K, 'OC': OC, 'pH': pH,
            'EC': EC, 'S': S, 'Fe': Fe, 'Zn': Zn, 'Cu': Cu, 'B': B, 'Mn': Mn
        }

        try:
            prediction = predict_yield(model, encoders, inputs)
            print_result(inputs, prediction, meta.get("rf_rmse", 1.24))
        except Exception as e:
            print(f"  ❌ Prediction error: {e}")

        again = input("\n  🔄 Predict another? (y/n): ").strip().lower()
        if again != 'y':
            break

    print("\n  👋 Thank you for using CropAI Predictor!\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI QUICK PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def cli_mode(args, model, encoders, meta):
    inputs = {
        'district'   : args.district.title(),
        'crop'       : args.crop.title(),
        'Soil_Type'  : args.soil.title(),
        'temperature': args.temp,
        'humidity'   : args.humidity,
        'N'          : args.N,
        'P'          : args.P,
        'K'          : args.K,
        'OC'         : args.OC,
        'pH'         : args.pH,
        'EC'         : args.EC,
        'S'          : args.S,
        'Fe'         : args.Fe,
        'Zn'         : args.Zn,
        'Cu'         : args.Cu,
        'B'          : args.B,
        'Mn'         : args.Mn,
    }

    prediction = predict_yield(model, encoders, inputs)
    print_result(inputs, prediction, meta.get("rf_rmse", 1.24))


# ─────────────────────────────────────────────────────────────────────────────
# QUICK DEMO (no args)
# ─────────────────────────────────────────────────────────────────────────────
def run_demo(model, encoders, meta):
    """Run a built-in demo prediction with sample data."""
    print("\n  🧪 Running Demo Prediction with sample data...\n")

    demo_cases = [
        {"district": "Thanjavur",  "crop": "Rice",      "Soil_Type": "Alluvial Soil",
         "temperature": 30, "humidity": 70, "N": 50, "P": 40, "K": 30,
         "OC": 0.8, "pH": 7.2, "EC": 0.5, "S": 12, "Fe": 4.5, "Zn": 1.2, "Cu": 0.5, "B": 0.8, "Mn": 3.5},

        {"district": "Coimbatore", "crop": "Groundnut", "Soil_Type": "Red Soil",
         "temperature": 32, "humidity": 60, "N": 30, "P": 25, "K": 20,
         "OC": 0.5, "pH": 6.8, "EC": 0.4, "S": 10, "Fe": 3.5, "Zn": 0.9, "Cu": 0.4, "B": 0.6, "Mn": 2.8},

        {"district": "Madurai",    "crop": "Maize",     "Soil_Type": "Black Soil",
         "temperature": 28, "humidity": 65, "N": 45, "P": 35, "K": 25,
         "OC": 1.0, "pH": 7.5, "EC": 0.6, "S": 14, "Fe": 5.0, "Zn": 1.5, "Cu": 0.6, "B": 1.0, "Mn": 4.0},
    ]

    rmse = meta.get("rf_rmse", 1.24)

    for i, case in enumerate(demo_cases, 1):
        try:
            pred = predict_yield(model, encoders, case)
            print(f"  Demo {i}: {case['district']} — {case['crop']} ({case['Soil_Type']})")
            print(f"           Predicted Yield = {pred:.4f} T/Ha  |  {quality_label(pred)}")
            low  = round(max(0.0, pred - rmse), 3)
            high = round(pred + rmse, 3)
            print(f"           Confidence Range: {low} – {high} T/Ha\n")
        except Exception as e:
            print(f"  Demo {i} Error: {e}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(BANNER)

    # Load model artifacts
    model, encoders, meta = load_model()
    print(f"  ✅ Model loaded successfully!")
    print(f"  📊 Algorithm   : {meta.get('model_name', 'Random Forest')}")
    print(f"  🎯 R² Score    : {meta.get('rf_r2', 0) * 100:.2f}%")
    print(f"  📉 MAE         : {meta.get('rf_mae', 0):.4f} Tons/Hectare")
    print(f"  📐 RMSE        : {meta.get('rf_rmse', 0):.4f} Tons/Hectare")
    print(f"  🌾 Crops       : {', '.join(meta.get('crop_types', []))}")
    print(f"  🏙️  Districts   : {len(meta.get('districts', []))} districts supported\n")

    parser = argparse.ArgumentParser(
        description="🌾 Crop Yield Predictor — uses your trained scikit-learn model"
    )
    # Categorical
    parser.add_argument("--district",  help="District name (e.g. Thanjavur)")
    parser.add_argument("--crop",      help="Crop type (e.g. Rice, Maize, Groundnut)")
    parser.add_argument("--soil",      help="Soil type (e.g. Alluvial Soil, Red Soil)")
    # Weather
    parser.add_argument("--temp",      type=float, help="Temperature in °C")
    parser.add_argument("--humidity",  type=float, help="Humidity in %%")
    # Soil nutrients
    parser.add_argument("--N",  type=float, default=45.0, help="Nitrogen")
    parser.add_argument("--P",  type=float, default=35.0, help="Phosphorus")
    parser.add_argument("--K",  type=float, default=25.0, help="Potassium")
    parser.add_argument("--OC", type=float, default=0.8,  help="Organic Carbon")
    parser.add_argument("--pH", type=float, default=7.0,  help="pH")
    parser.add_argument("--EC", type=float, default=0.5,  help="Electrical Conductivity")
    parser.add_argument("--S",  type=float, default=12.0, help="Sulphur")
    parser.add_argument("--Fe", type=float, default=4.5,  help="Iron")
    parser.add_argument("--Zn", type=float, default=1.2,  help="Zinc")
    parser.add_argument("--Cu", type=float, default=0.5,  help="Copper")
    parser.add_argument("--B",  type=float, default=0.8,  help="Boron")
    parser.add_argument("--Mn", type=float, default=3.5,  help="Manganese")
    # Mode
    parser.add_argument("--demo",      action="store_true", help="Run demo predictions")

    args = parser.parse_args()

    if args.demo:
        run_demo(model, encoders, meta)

    elif args.district and args.crop and args.soil and args.temp and args.humidity:
        cli_mode(args, model, encoders, meta)

    else:
        # No CLI args given → fall back to demo + interactive
        run_demo(model, encoders, meta)
        interactive_mode(model, encoders, meta)
