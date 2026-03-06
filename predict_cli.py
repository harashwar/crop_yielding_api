"""
╔══════════════════════════════════════════════════════════════════════╗
║   predict_cli.py — Command-Line Predictor                           ║
║   AI-Powered Crop Yield Prediction and Optimization                 ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
  python predict_cli.py
  python predict_cli.py --district Thanjavur --soil Loamy --humidity 65 --temp 29 --crop Rice
"""

import argparse
import pandas as pd
import joblib
import json
import sys
import os

MODEL_PATH    = "model/crop_yield_model.pkl"
ENCODERS_PATH = "model/label_encoders.pkl"
METADATA_PATH = "model/model_metadata.json"

VALID_DISTRICTS  = ["Thanjavur","Madurai","Coimbatore","Salem","Erode","Trichy",
                    "Dindigul","Tirunelveli","Kanyakumari","Namakkal","Karur",
                    "Vellore","Tiruppur","Cuddalore","Nagapattinam","Thoothukudi","Krishnagiri"]
VALID_SOIL_TYPES = ["Loamy","Clay","Sandy"]
VALID_CROPS      = ["Rice","Maize","Wheat","Millet","Groundnut","Coconut","Turmeric","Mango"]
FEATURE_COLS     = ['District', 'Soil_Type', 'Humidity', 'Temperature', 'Crop_Type']

BANNER = """
╔════════════════════════════════════════════════════╗
║  🌾  AI Crop Yield Predictor — CLI Mode           ║
║      Tamil Nadu Agriculture Intelligence System    ║
╚════════════════════════════════════════════════════╝
"""

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Please run: python train_model.py")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    return model, encoders, meta

def predict(model, encoders, district, soil_type, humidity, temperature, crop_type):
    encoded = {
        'District'   : encoders['District'].transform([district])[0],
        'Soil_Type'  : encoders['Soil_Type'].transform([soil_type])[0],
        'Humidity'   : float(humidity),
        'Temperature': float(temperature),
        'Crop_Type'  : encoders['Crop_Type'].transform([crop_type])[0],
    }
    df_in = pd.DataFrame([encoded], columns=FEATURE_COLS)
    return round(float(model.predict(df_in)[0]), 4)

def quality_label(yield_val):
    if yield_val >= 2.0: return "🌟 EXCELLENT"
    if yield_val >= 1.7: return "✅ GOOD"
    if yield_val >= 1.4: return "⚠️  AVERAGE"
    return "🔴 LOW"

def interactive_mode(model, encoders, meta):
    print("\n                  Interactive Prediction Mode")
    print("            (Type 'quit' at any time to exit)\n")

    while True:
        print("─" * 52)

        # District
        print(f"Available Districts: {', '.join(VALID_DISTRICTS)}")
        district = input("📍 Enter District: ").strip().title()
        if district.lower() == 'quit': break
        if district not in VALID_DISTRICTS:
            print(f"  ⚠️  Invalid district. Choose from the list above."); continue

        # Soil
        print(f"\nSoil types: {', '.join(VALID_SOIL_TYPES)}")
        soil_type = input("🪨 Enter Soil Type: ").strip().title()
        if soil_type.lower() == 'quit': break
        if soil_type not in VALID_SOIL_TYPES:
            print(f"  ⚠️  Invalid soil type. Choose from: {VALID_SOIL_TYPES}"); continue

        # Humidity
        try:
            humidity = float(input("💧 Enter Humidity (%): ").strip())
            if not (40 <= humidity <= 80):
                print("  ⚠️  Humidity should be between 40-80%."); continue
        except ValueError:
            print("  ⚠️  Invalid number."); continue

        # Temperature
        try:
            temperature = float(input("🌡️  Enter Temperature (°C): ").strip())
            if not (20 <= temperature <= 45):
                print("  ⚠️  Temperature should be between 20-45°C."); continue
        except ValueError:
            print("  ⚠️  Invalid number."); continue

        # Crop
        print(f"\nAvailable Crops: {', '.join(VALID_CROPS)}")
        crop_type = input("🌱 Enter Crop Type: ").strip().title()
        if crop_type.lower() == 'quit': break
        if crop_type not in VALID_CROPS:
            print(f"  ⚠️  Invalid crop. Choose from the list above."); continue

        # Predict
        try:
            result = predict(model, encoders, district, soil_type, humidity, temperature, crop_type)
            rmse   = meta.get("rf_rmse", 0.1)
            low    = round(max(0, result - rmse), 3)
            high   = round(result + rmse, 3)

            print("\n" + "═" * 52)
            print(f"  🌾 Predicted Yield : {result:.4f} Ton/Acre")
            print(f"  📊 Quality         : {quality_label(result)}")
            print(f"  📉 Confidence Range: {low} – {high} Ton/Acre")
            print("═" * 52)
        except Exception as e:
            print(f"  ❌ Prediction error: {e}")

        again = input("\n  🔄 Predict another? (y/n): ").strip().lower()
        if again != 'y':
            break

    print("\n  👋 Thank you for using CropAI Predictor!\n")

def cli_mode(args, model, encoders, meta):
    district    = args.district.title()
    soil_type   = args.soil.title()
    crop_type   = args.crop.title()
    humidity    = args.humidity
    temperature = args.temp

    if district not in VALID_DISTRICTS:
        print(f"❌ Invalid district: {district}"); sys.exit(1)
    if soil_type not in VALID_SOIL_TYPES:
        print(f"❌ Invalid soil type: {soil_type}"); sys.exit(1)
    if crop_type not in VALID_CROPS:
        print(f"❌ Invalid crop: {crop_type}"); sys.exit(1)

    result = predict(model, encoders, district, soil_type, humidity, temperature, crop_type)
    rmse   = meta.get("rf_rmse", 0.1)

    print(f"\n  Input  : {district} | {soil_type} | {humidity}% humidity | {temperature}°C | {crop_type}")
    print(f"  Yield  : {result:.4f} Ton/Acre")
    print(f"  Quality: {quality_label(result)}")
    print(f"  Range  : {max(0,round(result-rmse,3))} – {round(result+rmse,3)} Ton/Acre\n")

if __name__ == "__main__":
    print(BANNER)
    parser = argparse.ArgumentParser(description="Crop Yield Predictor CLI")
    parser.add_argument("--district", help="District name")
    parser.add_argument("--soil",     help="Soil type (Loamy/Clay/Sandy)")
    parser.add_argument("--humidity", type=float, help="Humidity %%")
    parser.add_argument("--temp",     type=float, help="Temperature °C")
    parser.add_argument("--crop",     help="Crop type")
    args = parser.parse_args()

    model, encoders, meta = load_artifacts()
    print(f"  ✅ Model loaded | R² = {meta.get('rf_r2',0)*100:.2f}% | Algorithm: {meta.get('model_name','RF')}\n")

    if args.district and args.soil and args.humidity and args.temp and args.crop:
        cli_mode(args, model, encoders, meta)
    else:
        interactive_mode(model, encoders, meta)
