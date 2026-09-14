"""
SageMaker inference handler for the SureGuard LightGBM model.

SageMaker's scikit-learn container calls these four functions:
  model_fn        -> load the model once when the endpoint starts
  input_fn        -> parse the incoming request body
  predict_fn      -> run the prediction
  output_fn       -> format the response body

Request format (JSON):
  {"instances": [[driver_age, vehicle_age, annual_mileage, prior_claims,
                  credit_score, region_risk, vehicle_power, exposure_months], ...]}

Response format (JSON):
  {"predictions": [expected_claims, ...]}
"""
import json
import os
import lightgbm as lgb

FEATURE_ORDER = [
    "driver_age", "vehicle_age", "annual_mileage", "prior_claims",
    "credit_score", "region_risk", "vehicle_power", "exposure_months",
]


def model_fn(model_dir):
    """Load the LightGBM booster from the model directory."""
    booster = lgb.Booster(model_file=os.path.join(model_dir, "model.txt"))
    return booster


def input_fn(request_body, request_content_type="application/json"):
    """Parse the request body into a list of feature rows."""
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")
    payload = json.loads(request_body)
    instances = payload.get("instances")
    if instances is None:
        raise ValueError('Request JSON must contain an "instances" key.')
    return instances


def predict_fn(input_data, model):
    """Run prediction. Returns expected monthly claim counts."""
    preds = model.predict(input_data)
    return [round(float(p), 4) for p in preds]


def output_fn(prediction, accept="application/json"):
    """Format the prediction as a JSON response."""
    return json.dumps({"predictions": prediction}), "application/json"
