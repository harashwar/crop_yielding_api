import pandas as pd, numpy as np, joblib, os, json, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('crop_yielding_predection.csv')
print('Loaded:', df.shape)
print('Columns:', df.columns.tolist())
print(df.head(3).to_string())

cat_cols = ['District','Soil_Type','Crop_Type']
label_encoders = {}
df_enc = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col])
    label_encoders[col] = le

FEAT = ['District','Soil_Type','Humidity','Temperature','Crop_Type']
X = df_enc[FEAT]
y = df_enc['Yield_Ton_per_Acre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train: {len(X_train)}  Test: {len(X_test)}')

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
mae  = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2   = r2_score(y_test, pred)
print(f'MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}')

cv = cross_val_score(rf, X, y, cv=5, scoring='r2')
print(f'CV R2: {cv.mean():.4f} +/- {cv.std():.4f}')

os.makedirs('model', exist_ok=True)
joblib.dump(rf, 'model/crop_yield_model.pkl')
joblib.dump(label_encoders, 'model/label_encoders.pkl')
meta = {
    'model_name': 'Random Forest',
    'features': FEAT,
    'target': 'Yield_Ton_per_Acre',
    'train_size': len(X_train),
    'test_size': len(X_test),
    'rf_mae': round(mae, 4),
    'rf_rmse': round(rmse, 4),
    'rf_r2': round(r2, 4),
    'gb_mae': round(mae, 4),
    'gb_rmse': round(rmse, 4),
    'gb_r2': round(r2, 4),
    'cv_mean_r2': round(float(cv.mean()), 4),
    'cv_std_r2': round(float(cv.std()), 4),
    'districts': sorted(df['District'].unique().tolist()),
    'soil_types': sorted(df['Soil_Type'].unique().tolist()),
    'crop_types': sorted(df['Crop_Type'].unique().tolist()),
    'humidity_range': [int(df['Humidity'].min()), int(df['Humidity'].max())],
    'temperature_range': [int(df['Temperature'].min()), int(df['Temperature'].max())]
}
with open('model/model_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print('Model + encoders + metadata saved successfully!')

# Sample predictions
samples = [
    ('Thanjavur',   'Loamy', 65, 29, 'Rice'),
    ('Coimbatore',  'Sandy', 52, 30, 'Maize'),
    ('Kanyakumari', 'Clay',  72, 26, 'Rice'),
    ('Tirunelveli', 'Sandy', 49, 35, 'Millet'),
]
print('\nSample Predictions:')
for d, s, h, t, c in samples:
    row = {
        'District':    label_encoders['District'].transform([d])[0],
        'Soil_Type':   label_encoders['Soil_Type'].transform([s])[0],
        'Humidity':    h,
        'Temperature': t,
        'Crop_Type':   label_encoders['Crop_Type'].transform([c])[0],
    }
    inp = pd.DataFrame([row], columns=FEAT)
    p = rf.predict(inp)[0]
    print(f'  {d:15} {s:6} H={h} T={t} {c:10} => {p:.4f} T/Acre')
