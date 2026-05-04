"""
train.py — SolarAI Model Training Pipeline
===========================================
Generates a realistic solar power dataset, applies the full ML pipeline,
and saves all model artefacts to model/.

Steps:
  1. Dataset generation  (physics-based, 6 000 hourly records)
  2. Outlier removal     (IQR, 1.5× rule)
  3. Feature engineering (cyclic time encoding, interaction terms)
  4. LDA                 (dimensionality reduction, 4-class binning)
  5. Model training      (SVR · Random Forest · Ridge Regression)
  6. Evaluation          (MAE · RMSE · R²)
  7. Save artefacts      → model/

Run:
    python train.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import joblib

warnings.filterwarnings("ignore")
os.makedirs("model", exist_ok=True)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  SolarAI — Model Training Pipeline")
print("=" * 60)
print("[1] Generating dataset …")

n = 6_000

hours    = np.random.randint(0, 24, n)
doys     = np.random.randint(1, 366, n)
months   = ((doys / 30.4).astype(int) % 12) + 1
irr_clear  = 900 * np.clip(np.sin(np.pi * (hours - 6) / 12), 0, 1)
cloud      = np.random.uniform(0.3, 1.0, n)
irradiance = np.clip(irr_clear * cloud + np.random.normal(0, 30, n), 0, 1100)
temperature = (30
               + 8 * np.sin(2 * np.pi * doys / 365)
               + 5 * np.sin(np.pi * (hours - 6) / 12)
               + np.random.normal(0, 3, n))
humidity   = np.clip(50 + 20 * cloud + np.random.normal(0, 8, n), 5, 100)
wind_speed = np.clip(np.random.exponential(3, n), 0, 20)
temp_coeff = np.clip(1 - 0.004 * np.clip(temperature - 25, 0, None), 0.6, 1.0)
hum_coeff  = np.clip(1 - 0.003 * humidity, 0.7, 1.0)
power_output = np.clip(
    (irradiance / 1000.0) * 50.0 * temp_coeff * hum_coeff + np.random.normal(0, 0.8, n),
    0, 60
)

df = pd.DataFrame({
    "hour": hours, "doy": doys, "month": months,
    "temperature": temperature, "irradiance": irradiance,
    "humidity": humidity, "wind_speed": wind_speed, "power_output": power_output,
})
print(f"     Generated {len(df):,} samples | "
      f"Power range: {df.power_output.min():.1f}–{df.power_output.max():.1f} kW")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  OUTLIER REMOVAL — IQR (1.5x)
# ─────────────────────────────────────────────────────────────────────────────

print("[2] Removing outliers (IQR 1.5x) …")
before = len(df)
outlier_stats = {}
for col in ["irradiance", "temperature", "wind_speed", "power_output"]:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr     = q3 - q1
    lo, hi  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    removed = int(((df[col] < lo) | (df[col] > hi)).sum())
    df      = df[(df[col] >= lo) & (df[col] <= hi)]
    outlier_stats[col] = {
        "q1": round(q1,2), "q3": round(q3,2), "iqr": round(iqr,2),
        "lower_bound": round(lo,2), "upper_bound": round(hi,2),
        "outliers_removed": removed,
    }
print(f"     {before:,} → {len(df):,} rows "
      f"({before - len(df):,} removed, {(before-len(df))/before*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

print("[3] Engineering features …")
df["hour_sin"]  = np.sin(2 * np.pi * df.hour  / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df.hour  / 24)
df["doy_sin"]   = np.sin(2 * np.pi * df.doy   / 365)
df["doy_cos"]   = np.cos(2 * np.pi * df.doy   / 365)
df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.month / 12)
df["irr_temp"]  = df.irradiance * df.temperature / 1000
df["irr_hum"]   = df.irradiance * (1 - df.humidity / 100)
df["irr_norm"]  = df.irradiance / 1000.0

FEATURES = [
    "temperature", "irradiance", "humidity", "wind_speed",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "month_sin", "month_cos", "irr_temp", "irr_hum", "irr_norm",
]
TARGET = "power_output"
print(f"     {len(FEATURES)} features created")

df      = df.reset_index(drop=True)
split   = int(len(df) * 0.80)
X_tr    = df[FEATURES].values[:split];  X_te = df[FEATURES].values[split:]
y_tr    = df[TARGET].values[:split];    y_te = df[TARGET].values[split:]

# ─────────────────────────────────────────────────────────────────────────────
# 4.  SCALING + LDA
# ─────────────────────────────────────────────────────────────────────────────

print("[4] Scaling + LDA …")
scaler   = StandardScaler()
X_tr_s   = scaler.fit_transform(X_tr)
X_te_s   = scaler.transform(X_te)

# Bin power into 4 classes for LDA supervision
y_class  = pd.cut(pd.Series(y_tr.astype(float)),
                  bins=[-0.001, 10, 25, 40, 200], labels=[0,1,2,3],
                  include_lowest=True).astype(int)
n_comp   = min(3, int(y_class.nunique()) - 1)
lda      = LinearDiscriminantAnalysis(n_components=n_comp)
lda.fit(X_tr_s, y_class)
X_tr_l   = lda.transform(X_tr_s)
X_te_l   = lda.transform(X_te_s)
print(f"     {len(FEATURES)} features → {lda.n_components} LDA components  "
      f"expl_var={lda.explained_variance_ratio_.round(3).tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(name, y_true, y_pred):
    yp   = np.clip(y_pred, 0, 60)
    mae  = mean_absolute_error(y_true, yp)
    rmse = np.sqrt(mean_squared_error(y_true, yp))
    r2   = r2_score(y_true, yp)
    print(f"     {name:<34} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")
    return {"mae": round(mae,3), "rmse": round(rmse,3), "r2": round(r2,4)}

print("[5a] Training SVR (RBF) …")
svr = SVR(kernel="rbf", C=50, gamma="scale", epsilon=0.5)
svr.fit(X_tr_l, y_tr)
svm_pred    = svr.predict(X_te_l)
svm_metrics = evaluate("SVM (SVR)", y_te, svm_pred)

print("[5b] Training Bagging Ensemble (Random Forest) …")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=3,
                           max_features=0.7, random_state=42, n_jobs=-1)
rf.fit(X_tr_l, y_tr)
bag_pred    = rf.predict(X_te_l)
bag_metrics = evaluate("Bagging Ensemble (RF)", y_te, bag_pred)

print("[5c] Training Time-Series Regression (Ridge) …")
ridge = Ridge(alpha=1.0)
ridge.fit(X_tr_l, y_tr)
ts_pred    = ridge.predict(X_te_l)
ts_metrics = evaluate("Time-Series Regression (Ridge)", y_te, ts_pred)

ens_pred = np.clip(0.25*svm_pred + 0.50*bag_pred + 0.25*ts_pred, 0, 60)
ens_r2   = r2_score(y_te, ens_pred)
print(f"     {'Weighted Ensemble':<34} R2={ens_r2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

print("[6] Computing feature importance …")
n_lda    = lda.n_components
rf_imp   = rf.feature_importances_[:n_lda]
orig_imp = np.abs(lda.scalings_[:, :n_lda]) @ rf_imp
orig_imp = orig_imp / orig_imp.sum()

_rename = {
    "temperature":"Temperature", "irradiance":"Solar Irradiance",
    "humidity":"Humidity", "wind_speed":"Wind Speed",
    "hour_sin":"Hour of Day", "hour_cos":"Hour of Day",
    "doy_sin":"Day of Year",  "doy_cos":"Day of Year",
    "month_sin":"Month",      "month_cos":"Month",
    "irr_temp":"Irradiance×Temperature", "irr_hum":"Irradiance×Humidity",
    "irr_norm":"Normalised Irradiance",
}
merged = {}
for fname, imp in zip(FEATURES, orig_imp):
    nice = _rename.get(fname, fname)
    merged[nice] = merged.get(nice, 0.0) + imp
total   = sum(merged.values())
fi_items = sorted(
    [{"feature": k, "importance": round(v/total, 4)} for k,v in merged.items()],
    key=lambda x: -x["importance"],
)[:5]
print(f"     Top: {[i['feature'] for i in fi_items]}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  TIME-SERIES DATA (actual vs predicted on held-out test set)
# ─────────────────────────────────────────────────────────────────────────────

print("[7] Generating time-series export (168 points) …")
N = min(168, len(X_te))
ts_points = []
for i in range(N):
    ts = (datetime(2024, 1, 1) + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ts_points.append({
        "timestamp":           ts,
        "actual":              round(float(y_te[i]), 2),
        "svmPredicted":        round(float(np.clip(svm_pred[i],  0, 60)), 2),
        "baggingPredicted":    round(float(np.clip(bag_pred[i],  0, 60)), 2),
        "timeSeriesPredicted": round(float(np.clip(ts_pred[i],   0, 60)), 2),
    })

# ─────────────────────────────────────────────────────────────────────────────
# 8.  CLUSTERING (K-Means k=4)
# ─────────────────────────────────────────────────────────────────────────────

print("[8] K-Means clustering (k=4) …")
SITES = ["Rajasthan","Jodhpur","Jaisalmer","Gujarat","Haryana",
         "Punjab","Tamil Nadu","Kerala","Andhra Pradesh","Maharashtra","Odisha","West Bengal"]
CLUSTER_LABELS = [
    "High Irradiance — Arid", "Moderate — Temperate",
    "Low — Coastal / Humid",  "Variable — Monsoon",
]
km = KMeans(n_clusters=4, random_state=42, n_init=10)
km.fit(df[["irradiance", "temperature"]])
df["cluster"] = km.labels_
order = np.argsort(km.cluster_centers_[:, 0])[::-1]
remap = {old: new for new, old in enumerate(order)}
df["cluster"] = df["cluster"].map(remap)
centroids = km.cluster_centers_[order]

cluster_pts = []
for ci in range(4):
    sub = df[df.cluster == ci].sample(min(75, int((df.cluster==ci).sum())), random_state=42)
    for idx, row in sub.iterrows():
        cluster_pts.append({
            "irradiance":  round(float(row.irradiance), 1),
            "temperature": round(float(row.temperature), 1),
            "power":       round(float(row.power_output), 1),
            "cluster":     int(row.cluster),
            "site":        SITES[int(idx) % len(SITES)],
        })
centroid_list = [
    {"cluster": i, "irradiance": round(float(centroids[i,0]),1),
     "temperature": round(float(centroids[i,1]),1), "label": CLUSTER_LABELS[i]}
    for i in range(4)
]
for i in range(4):
    print(f"     C{i} {CLUSTER_LABELS[i]}: {int((df.cluster==i).sum())} samples")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

best_r2 = max(svm_metrics["r2"], bag_metrics["r2"], ts_metrics["r2"])
summary = {
    "totalPredictions":  len(df),
    "avgDailyOutput":    round(float(df.power_output.mean()), 1),
    "bestModelAccuracy": round(best_r2 * 100, 1),
    "sitesMonitored":    len(SITES),
    "peakPower":         round(float(df.power_output.max()), 1),
    "avgConfidence":     round(float(ens_r2), 3),
}

# ─────────────────────────────────────────────────────────────────────────────
# 10. SAVE ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

print("[9] Saving artefacts to model/ …")
joblib.dump(svr,      "model/svr_model.pkl")
joblib.dump(rf,       "model/rf_model.pkl")
joblib.dump(ridge,    "model/ridge_model.pkl")
joblib.dump(scaler,   "model/scaler.pkl")
joblib.dump(lda,      "model/lda.pkl")
joblib.dump(km,       "model/kmeans.pkl")
joblib.dump(FEATURES, "model/feature_names.pkl")

json.dump({"svm": svm_metrics, "bagging": bag_metrics, "timeseries": ts_metrics},
          open("model/metrics.json", "w"), indent=2)
json.dump({"items": fi_items},
          open("model/feature_importance.json", "w"), indent=2)
json.dump({"points": ts_points},
          open("model/timeseries_data.json", "w"), indent=2)
json.dump({"points": cluster_pts, "centroids": centroid_list, "numClusters": 4},
          open("model/cluster_data.json", "w"), indent=2)
json.dump(outlier_stats,
          open("model/outlier_stats.json", "w"), indent=2)
json.dump(summary,
          open("model/summary.json", "w"), indent=2)

print()
print("=" * 60)
print("  Training Complete!")
print("=" * 60)
print(f"  SVM R²          : {svm_metrics['r2']:.4f}")
print(f"  Bagging R²      : {bag_metrics['r2']:.4f}")
print(f"  Time-Series R²  : {ts_metrics['r2']:.4f}")
print(f"  Ensemble R²     : {ens_r2:.4f}")
print()
print("  Next step: python app.py → http://localhost:5000")
print("=" * 60)