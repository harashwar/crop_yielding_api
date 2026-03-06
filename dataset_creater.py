import json
import random
import pandas as pd

def calculate_yield(row):
    crop = row["crop"]
    # Base yield in tons per hectare
    base_yields = {
        "Rice": 4.5,
        "Maize": 3.0,
        "Millet": 1.5,
        "Sugarcane": 60.0,
        "Groundnut": 1.2,
        "Mango": 8.0,
        "Guava": 10.0,
        "Coconut": 5.0
    }
    base = base_yields.get(crop, 3.0)
    
    # Simple environmental and soil factors
    ph_factor = 1.0 - abs(row["pH"] - 7.0) * 0.1
    
    # NPK impact
    npk_factor = (row["N"] + row["P"] + row["K"]) / 150.0
    npk_factor = max(0.5, min(1.5, npk_factor))
    
    # Weather impact
    temp_factor = 1.0 - abs(row["temperature"] - 30) * 0.02
    hum_factor = 1.0 - abs(row["humidity"] - 60) * 0.01
    
    final_yield = base * ph_factor * npk_factor * temp_factor * hum_factor
    
    # Add random variation
    final_yield *= random.uniform(0.9, 1.1)
    
    return round(max(0.5, final_yield), 2)

# Load your geojson file
with open("soil_nutrients_dataset.geojson") as f:
    data = json.load(f)

rows = []

for feature in data["features"]:
    p = feature["properties"]

    district = p["district"]

    # convert base values
    N = int(p["N_percentage"])
    P = int(p["P_percentage"])
    K = int(p["K_percentage"])
    OC = int(p["OC_percentage"])
    PH = int(p["pH_percentage"])
    EC = int(p["EC_percentage"])
    S = int(p["S_percentage"])
    FE = int(p["Fe_percentage"])
    ZN = int(p["Zn_percentage"])
    CU = int(p["Cu_percentage"])
    B = int(p["B_percentage"])
    MN = int(p["Mn_percentage"])
    
    # Simple mapping of Soil_Type based on District or Randomly assign
    # For a synthetic dataset, we assign random soil types if not available in geojson
    soil_types_available = ["Loamy", "Clay", "Sandy","Red Soil","Black Soil","Alluvial Soil","Laterite Soil","Peaty Soil"]

    # generate multiple samples per district ensuring EVERY soil type is included
    for i in range(8):   # 8 samples per soil type per district
        for soil in soil_types_available:
            
            row = {
                "district": district,

                "N": max(0, min(100, N + random.randint(-10,10))),
                "P": max(0, min(100, P + random.randint(-10,10))),
                "K": max(0, min(100, K + random.randint(-10,10))),
                "OC": max(0, min(100, OC + random.randint(-10,10))),

                "pH": round(random.uniform(6.5,8.5),2),

                "EC": max(0, min(100, EC + random.randint(-10,10))),
                "S": max(0, min(100, S + random.randint(-10,10))),

                "Fe": max(0, min(100, FE + random.randint(-10,10))),
                "Zn": max(0, min(100, ZN + random.randint(-10,10))),
                "Cu": max(0, min(100, CU + random.randint(-10,10))),
                "B": max(0, min(100, B + random.randint(-10,10))),
                "Mn": max(0, min(100, MN + random.randint(-10,10))),

                "temperature": random.randint(24,35),
                "humidity": random.randint(45,80),

                "crop": random.choice([
                    "Rice","Maize","Millet","Sugarcane","Groundnut",
                    "Mango","Guava","Coconut"
                ]),
                
                "Soil_Type": soil
            }

            row["Yield_tons_per_hectare"] = calculate_yield(row)
            rows.append(row)

df = pd.DataFrame(rows)

df.to_csv("soil_dataset_2000_with_yield.csv",index=False)

print("Dataset created: soil_dataset_2000_with_yield.csv")
print("Total rows:",len(df))