import pandas as pd
import numpy as np
import pygeohash as pgh
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# =========================
# Load Data
# =========================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Save IDs
test_ids = test["Index"]

# =========================
# Feature Engineering
# =========================

def create_features(df):
    df = df.copy()

    # Decode geohash into coordinates

    df["lat"] = df["geohash"].apply(
        lambda x: pgh.decode(x)[0]
    )

    df["lon"] = df["geohash"].apply(
        lambda x: pgh.decode(x)[1]
    )
    t = df["timestamp"].astype(str).str.split(":", expand=True)

    df["hour"] = t[0].astype(int)
    df["minute"] = t[1].astype(int)
    df["time_float"] = df["hour"] + df["minute"] / 60

    df["hour_sin"] = np.sin(2 * np.pi * df["time_float"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time_float"] / 24)
    
    df["lat_hour"] = df["lat"] * df["hour"]
    df["lon_hour"] = df["lon"] * df["hour"]

    df["lat_temp"] = (
        df["lat"] *
        df["Temperature"].fillna(0)
    )

    df["lon_temp"] = (
        df["lon"] *
        df["Temperature"].fillna(0)
    )

    df["morning_peak"] = ((df["hour"] >= 7) & (df["hour"] <= 10)).astype(int)
    df["evening_peak"] = ((df["hour"] >= 17) & (df["hour"] <= 20)).astype(int)
    
    df["rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 10))
        |
        ((df["hour"] >= 16) & (df["hour"] <= 20))
    ).astype(int)

    # IMPORTANT
    df.drop(columns=["timestamp"], inplace=True)

    return df

train = create_features(train)
test = create_features(test)

# =========================
# Missing Values
# =========================

for col in ["Temperature"]:
    med = train[col].median()

    train[col] = train[col].fillna(med)
    test[col] = test[col].fillna(med)

for col in ["RoadType", "Weather"]:
    mode = train[col].mode()[0]

    train[col] = train[col].fillna(mode)
    test[col] = test[col].fillna(mode)

# =========================
# Prepare Data
# =========================

TARGET = "demand"

drop_cols = ["Index", TARGET]

X = train.drop(columns=drop_cols)
y = train[TARGET]

X_test = test.drop(columns=["Index"])

cat_features = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "day"
]

# =========================
# Cross Validation
# =========================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]
    
    # =========================
    # For CPU Training
    # =========================

#    model = CatBoostRegressor(
#        iterations=5000,
#        depth=10,
#        learning_rate=0.03,
#        loss_function="RMSE",
#        eval_metric="R2",
#        random_seed=42,
#        verbose=500
#    )

    # =========================
    # For GPU Training
    # =========================

    model = CatBoostRegressor(
        iterations=5000,
        depth=10,
        learning_rate=0.03,
        loss_function="RMSE",
        eval_metric="R2",
        task_type="GPU",
        devices="0",
        random_seed=42,
        verbose=500
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )

    preds = model.predict(X_valid)

    fold_r2 = r2_score(y_valid, preds)

    print(f"Fold {fold+1} R2 = {fold_r2:.5f}")

    oof[valid_idx] = preds

    test_preds += model.predict(X_test) / kf.n_splits

print("\nCV R2 =", r2_score(y, oof))

# =========================
# Submission
# =========================

submission = pd.DataFrame({
    "Index": test_ids,
    "demand": test_preds
})

submission.to_csv("submission.csv", index=False)

print("\nSubmission Saved")
print(submission.head())
print(submission.shape)