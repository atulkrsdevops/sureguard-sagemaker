"""
SureGuard - train a LightGBM claim-frequency model (Poisson objective).

Regenerates a model consistent with the SureGuard project:
  - predicts monthly claim counts for an insurance portfolio
  - LightGBM with Poisson objective
  - realistic tabular features (policy + exposure + risk signals)

Outputs:
  build/model.txt        -> the raw LightGBM booster
  build/feature_names.json
  build/sample_input.json -> a ready-to-use test payload
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

RNG = np.random.default_rng(42)
N = 40_000
BUILD = Path(__file__).parent / "build"
BUILD.mkdir(exist_ok=True)

# ---- 1. Synthesize an insurance portfolio dataset -------------------------
# Features an actuary would actually use for claim frequency.
driver_age      = RNG.integers(18, 80, N)
vehicle_age     = RNG.integers(0, 20, N)
annual_mileage  = RNG.normal(12_000, 4_000, N).clip(1_000, 40_000)
prior_claims    = RNG.poisson(0.3, N)
credit_score    = RNG.normal(680, 70, N).clip(300, 850)
region_risk     = RNG.integers(1, 6, N)          # 1 low ... 5 high
vehicle_power   = RNG.normal(120, 40, N).clip(50, 400)  # hp
exposure_months = RNG.integers(1, 13, N)          # months the policy was active

# ---- 2. Build a "true" latent risk -> Poisson claim counts ----------------
# Log-rate driven by the features (this is what the model must recover).
log_rate = (
    -3.2
    + 0.015 * (driver_age < 25) * (25 - driver_age)     # young drivers riskier
    + 0.010 * (driver_age > 70) * (driver_age - 70)     # very old drivers riskier
    + 0.03  * vehicle_age
    + 0.00003 * annual_mileage
    + 0.35  * prior_claims
    - 0.002 * (credit_score - 680)
    + 0.18  * region_risk
    + 0.002 * (vehicle_power - 120)
)
# exposure acts as the Poisson "offset": more active months -> more expected claims
expected = np.exp(log_rate) * exposure_months
claims = RNG.poisson(expected)

df = pd.DataFrame({
    "driver_age": driver_age,
    "vehicle_age": vehicle_age,
    "annual_mileage": annual_mileage.round(0),
    "prior_claims": prior_claims,
    "credit_score": credit_score.round(0),
    "region_risk": region_risk,
    "vehicle_power": vehicle_power.round(0),
    "exposure_months": exposure_months,
    "claims": claims,
})

FEATURES = [c for c in df.columns if c != "claims"]

# ---- 3. Train/valid split -------------------------------------------------
mask = RNG.random(N) < 0.8
train, valid = df[mask], df[~mask]

train_set = lgb.Dataset(train[FEATURES], label=train["claims"])
valid_set = lgb.Dataset(valid[FEATURES], label=valid["claims"], reference=train_set)

params = {
    "objective": "poisson",       # claim COUNTS -> Poisson, as in SureGuard
    "metric": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
}

booster = lgb.train(
    params,
    train_set,
    num_boost_round=400,
    valid_sets=[valid_set],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
)

# ---- 4. Save artifacts ----------------------------------------------------
booster.save_model(str(BUILD / "model.txt"))
(BUILD / "feature_names.json").write_text(json.dumps(FEATURES, indent=2))

# a realistic sample request (one policy) for later endpoint testing
sample = {"instances": [valid[FEATURES].iloc[0].tolist()]}
(BUILD / "sample_input.json").write_text(json.dumps(sample, indent=2))

# quick sanity check
pred = booster.predict(valid[FEATURES].iloc[:5])
print("\nSanity — predicted expected claims for 5 policies:", pred.round(3).tolist())
print("Actual claims for those 5 policies:            ", valid["claims"].iloc[:5].tolist())
print("\nFeatures:", FEATURES)
print("Saved: model.txt, feature_names.json, sample_input.json")
