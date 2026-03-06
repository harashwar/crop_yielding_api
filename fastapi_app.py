"""
╔══════════════════════════════════════════════════════════════════════╗
║   AI-Powered Crop Yield Prediction — FastAPI Backend                ║
║   fastapi_app.py                                                    ║
║                                                                      ║
║   Run:  uvicorn fastapi_app:app --reload --port 8000                ║
║   Docs: http://localhost:8000/docs   (auto Swagger UI)              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import requests as http_requests
warnings.filterwarnings("ignore")

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "model", "crop_yield_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "model", "label_encoders.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model", "model_metadata.json")
GEOJSON_PATH  = os.path.join(BASE_DIR, "soil_nutrients_dataset.geojson")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS (must exactly match training order)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "district", "N", "P", "K", "OC", "pH", "EC", "S",
    "Fe", "Zn", "Cu", "B", "Mn", "temperature", "humidity", "crop", "Soil_Type"
]

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY STORE (populated on startup)
# ─────────────────────────────────────────────────────────────────────────────
store = {
    "model":          None,
    "label_encoders": None,
    "metadata":       None,
    "soil_data":      {},   # district → nutrient dict
}


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — load model once at startup
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artifacts when the server starts; clean up on shutdown."""
    print("🚀 Loading model artifacts...")

    # 1. Model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"❌ Model not found at '{MODEL_PATH}'. Run: python train_model.py"
        )
    store["model"] = joblib.load(MODEL_PATH)
    print(f"  ✅ Model loaded  : {MODEL_PATH}")

    # 2. Label Encoders
    if not os.path.exists(ENCODERS_PATH):
        raise RuntimeError(f"❌ Encoders not found at '{ENCODERS_PATH}'.")
    store["label_encoders"] = joblib.load(ENCODERS_PATH)
    print(f"  ✅ Encoders loaded: {ENCODERS_PATH}")

    # 3. Metadata
    if not os.path.exists(METADATA_PATH):
        raise RuntimeError(f"❌ Metadata not found at '{METADATA_PATH}'.")
    with open(METADATA_PATH, "r") as f:
        store["metadata"] = json.load(f)
    meta = store["metadata"]
    print(f"  ✅ Metadata loaded : R²={meta.get('rf_r2',0)*100:.2f}%  MAE={meta.get('rf_mae',0):.4f}")

    # 4. Soil nutrient GeoJSON (optional — used for auto-fill)
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH, "r") as f:
            geojson = json.load(f)
        for feature in geojson.get("features", []):
            p    = feature.get("properties", {})
            dist = p.get("district")
            if dist:
                store["soil_data"][dist] = {
                    "N":  float(p.get("N_percentage",  0)),
                    "P":  float(p.get("P_percentage",  0)),
                    "K":  float(p.get("K_percentage",  0)),
                    "OC": float(p.get("OC_percentage", 0)),
                    "pH": float(p.get("pH_percentage", 0)),
                    "EC": float(p.get("EC_percentage", 0)),
                    "S":  float(p.get("S_percentage",  0)),
                    "Fe": float(p.get("Fe_percentage", 0)),
                    "Zn": float(p.get("Zn_percentage", 0)),
                    "Cu": float(p.get("Cu_percentage", 0)),
                    "B":  float(p.get("B_percentage",  0)),
                    "Mn": float(p.get("Mn_percentage", 0)),
                }
        print(f"  ✅ Soil data loaded: {len(store['soil_data'])} districts")
    else:
        print("  ⚠️  GeoJSON soil file not found — manual nutrient input required.")

    print("\n🌾 Crop Yield Prediction API is ready!\n")
    yield
    # Shutdown — nothing to release for pickle models
    print("👋 Server shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "🌾 Crop Yield Prediction API",
    description = (
        "AI-powered crop yield prediction for Tamil Nadu districts. "
        "Uses a Random Forest model trained on soil nutrients, weather, and crop data."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# Allow frontend / browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS (request / response)
# ─────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Input payload for /predict endpoint."""

    # Required — categorical
    district  : str = Field(..., example="Thanjavur",    description="Tamil Nadu district name")
    crop      : str = Field(..., example="Rice",          description="Crop type")
    Soil_Type : str = Field(..., example="Alluvial Soil", description="Soil type")

    # Required — weather
    temperature : float = Field(..., ge=10, le=50,   example=30.0, description="Temperature in °C")
    humidity    : float = Field(..., ge=10, le=100,  example=70.0, description="Humidity in %")

    # Soil nutrients — optional, auto-filled from GeoJSON if not provided
    N  : Optional[float] = Field(None, ge=0, description="Nitrogen (kg/ha)")
    P  : Optional[float] = Field(None, ge=0, description="Phosphorus (kg/ha)")
    K  : Optional[float] = Field(None, ge=0, description="Potassium (kg/ha)")
    OC : Optional[float] = Field(None, ge=0, description="Organic Carbon (%)")
    pH : Optional[float] = Field(None, ge=0, le=14, description="Soil pH")
    EC : Optional[float] = Field(None, ge=0, description="Electrical Conductivity (dS/m)")
    S  : Optional[float] = Field(None, ge=0, description="Sulphur (mg/kg)")
    Fe : Optional[float] = Field(None, ge=0, description="Iron (mg/kg)")
    Zn : Optional[float] = Field(None, ge=0, description="Zinc (mg/kg)")
    Cu : Optional[float] = Field(None, ge=0, description="Copper (mg/kg)")
    B  : Optional[float] = Field(None, ge=0, description="Boron (mg/kg)")
    Mn : Optional[float] = Field(None, ge=0, description="Manganese (mg/kg)")

    @validator("district", "crop", "Soil_Type", pre=True)
    def title_case(cls, v):
        return str(v).strip().title()


class PredictResponse(BaseModel):
    predicted_yield  : float
    unit             : str
    quality_label    : str
    quality_emoji    : str
    confidence_low   : float
    confidence_high  : float
    inputs_used      : dict
    model_info       : dict


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _quality(yield_val: float) -> tuple[str, str]:
    if yield_val >= 8.0: return "Excellent", "🌟"
    if yield_val >= 5.0: return "Good",      "✅"
    if yield_val >= 2.5: return "Average",   "⚠️"
    return "Low", "🔴"


def _encode_and_predict(inputs: dict) -> float:
    """
    Encode categoricals with saved LabelEncoders and run the RF model.
    Returns predicted yield (float, Tons/Hectare).
    """
    encoders = store["label_encoders"]
    model    = store["model"]
    row      = inputs.copy()

    for cat in ("district", "crop", "Soil_Type"):
        encoder = encoders[cat]
        value   = str(row[cat])
        if value not in encoder.classes_:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{value}' is not a recognised value for '{cat}'. "
                    f"Valid options: {list(encoder.classes_)}"
                )
            )
        row[cat] = int(encoder.transform([value])[0])

    df  = pd.DataFrame([row], columns=FEATURE_COLS)
    return round(float(model.predict(df)[0]), 4)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def home():
    """Root — check the API is running."""
    return {
        "message"    : "🌾 Crop Yield Prediction API is running!",
        "docs"       : "/docs",
        "redoc"      : "/redoc",
        "health"     : "/health",
        "predict"    : "POST /predict",
        "metadata"   : "/metadata",
    }


@app.get("/health", tags=["General"])
def health():
    """Health check — confirms model is loaded."""
    return {
        "status"       : "ok",
        "model_loaded" : store["model"] is not None,
        "encoders_ok"  : store["label_encoders"] is not None,
        "soil_districts": len(store["soil_data"]),
    }


@app.get("/metadata", tags=["General"])
def get_metadata():
    """Return model metadata, supported districts, crops, and soil types."""
    if store["metadata"] is None:
        raise HTTPException(status_code=503, detail="Model metadata not available.")
    return store["metadata"]


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(payload: PredictRequest):
    """
    **Predict crop yield** given district, crop, soil type, and weather.

    - Soil nutrients (N, P, K, …) are **optional** — if omitted, they are
      auto-filled from the district's GeoJSON profile.
    - All categorical values are validated against training labels.
    """
    if store["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please restart the server.")

    meta = store["metadata"]

    # ── Resolve soil nutrients ─────────────────────────────────────────────
    soil_defaults = store["soil_data"].get(payload.district, {})

    nutrients = {
        "N":  payload.N  if payload.N  is not None else soil_defaults.get("N",  45.0),
        "P":  payload.P  if payload.P  is not None else soil_defaults.get("P",  35.0),
        "K":  payload.K  if payload.K  is not None else soil_defaults.get("K",  25.0),
        "OC": payload.OC if payload.OC is not None else soil_defaults.get("OC", 0.8),
        "pH": payload.pH if payload.pH is not None else soil_defaults.get("pH", 7.0),
        "EC": payload.EC if payload.EC is not None else soil_defaults.get("EC", 0.5),
        "S":  payload.S  if payload.S  is not None else soil_defaults.get("S",  12.0),
        "Fe": payload.Fe if payload.Fe is not None else soil_defaults.get("Fe", 4.5),
        "Zn": payload.Zn if payload.Zn is not None else soil_defaults.get("Zn", 1.2),
        "Cu": payload.Cu if payload.Cu is not None else soil_defaults.get("Cu", 0.5),
        "B":  payload.B  if payload.B  is not None else soil_defaults.get("B",  0.8),
        "Mn": payload.Mn if payload.Mn is not None else soil_defaults.get("Mn", 3.5),
    }

    # ── Build feature dict in exact training column order ──────────────────
    inputs = {
        "district"   : payload.district,
        "N"          : nutrients["N"],
        "P"          : nutrients["P"],
        "K"          : nutrients["K"],
        "OC"         : nutrients["OC"],
        "pH"         : nutrients["pH"],
        "EC"         : nutrients["EC"],
        "S"          : nutrients["S"],
        "Fe"         : nutrients["Fe"],
        "Zn"         : nutrients["Zn"],
        "Cu"         : nutrients["Cu"],
        "B"          : nutrients["B"],
        "Mn"         : nutrients["Mn"],
        "temperature": payload.temperature,
        "humidity"   : payload.humidity,
        "crop"       : payload.crop,
        "Soil_Type"  : payload.Soil_Type,
    }

    # ── Predict ────────────────────────────────────────────────────────────
    prediction = _encode_and_predict(inputs)

    rmse    = meta.get("rf_rmse", 1.24)
    quality, emoji = _quality(prediction)

    return PredictResponse(
        predicted_yield  = prediction,
        unit             = "Tons/Hectare",
        quality_label    = quality,
        quality_emoji    = emoji,
        confidence_low   = round(max(0.0, prediction - rmse), 3),
        confidence_high  = round(prediction + rmse, 3),
        inputs_used      = {**inputs, "nutrients_source": "geojson" if not payload.N else "manual"},
        model_info       = {
            "algorithm" : meta.get("model_name", "Random Forest"),
            "r2_score"  : meta.get("rf_r2", 0),
            "mae"       : meta.get("rf_mae", 0),
            "rmse"      : rmse,
        },
    )


@app.get("/predict/demo", tags=["Prediction"])
def predict_demo():
    """
    Run 3 built-in demo predictions — useful for testing without a client.
    """
    demo_cases = [
        PredictRequest(district="Thanjavur",  crop="Rice",      Soil_Type="Alluvial Soil", temperature=30, humidity=70),
        PredictRequest(district="Coimbatore", crop="Groundnut", Soil_Type="Red Soil",      temperature=32, humidity=60),
        PredictRequest(district="Madurai",    crop="Maize",     Soil_Type="Black Soil",    temperature=28, humidity=65),
    ]
    results = []
    for case in demo_cases:
        try:
            result = predict(case)
            results.append({
                "district": case.district,
                "crop"    : case.crop,
                "soil"    : case.Soil_Type,
                "yield"   : result.predicted_yield,
                "quality" : f"{result.quality_emoji} {result.quality_label}",
                "range"   : f"{result.confidence_low} – {result.confidence_high} T/Ha",
            })
        except Exception as e:
            results.append({"district": case.district, "error": str(e)})
    return {"demo_results": results}


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
