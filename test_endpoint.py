"""
Invoke the deployed SureGuard endpoint with a sample policy and print the
predicted expected monthly claim count.

Feature order:
  [driver_age, vehicle_age, annual_mileage, prior_claims,
   credit_score, region_risk, vehicle_power, exposure_months]
"""
import json
import boto3

REGION = "us-east-2"
ENDPOINT = "sureguard-endpoint"

runtime = boto3.client("sagemaker-runtime", region_name=REGION)

# Two example policies:
#   - a low-risk older driver, clean history
#   - a high-risk young driver, prior claims, high-risk region
payload = {
    "instances": [
        [55, 3, 9000, 0, 780, 1, 110, 12],   # low risk
        [21, 12, 20000, 2, 580, 5, 250, 12], # high risk
    ]
}

resp = runtime.invoke_endpoint(
    EndpointName=ENDPOINT,
    ContentType="application/json",
    Body=json.dumps(payload),
)

result = json.loads(resp["Body"].read())
print("Predicted expected monthly claims:")
for row, pred in zip(payload["instances"], result["predictions"]):
    print(f"  policy {row} -> {pred}")
