# SureGuard → SageMaker Serverless Deployment

Deploys the SureGuard LightGBM claim-frequency model to a SageMaker
**Serverless Inference** endpoint in `us-east-2`. No hourly cost — bills only
per request and scales to zero when idle.

## What's in here

| File | Purpose |
|------|---------|
| `train_model.py` | Regenerates & trains the LightGBM (Poisson) model → `build/model.txt` |
| `model.tar.gz` | **Pre-built** model artifact, ready to deploy (model + inference code) |
| `code/inference.py` | SageMaker handler (load / parse / predict / format) |
| `code/requirements.txt` | Installs LightGBM into the container |
| `deploy.py` | Uploads to S3 + creates the serverless endpoint |
| `test_endpoint.py` | Invokes the endpoint with two sample policies |
| `cleanup.py` | Deletes endpoint / config / model |

`model.tar.gz` is already built, so you can skip straight to deploying.
Re-run `train_model.py` only if you want to regenerate the model.

## One-time prerequisites

```bash
pip install sagemaker boto3
aws configure          # region = us-east-2, IAM creds with SageMaker access
```

You also need a **SageMaker execution role ARN** — the role SageMaker assumes to
read your model from S3. If you already opened SageMaker Studio / a domain, one
exists at:
`arn:aws:iam::<account-id>:role/service-role/AmazonSageMaker-ExecutionRole-XXXX`
(find it in IAM → Roles → search "SageMaker"). Paste it into `deploy.py`.

## Deploy (3 commands)

```bash
# 1. edit deploy.py -> set ROLE_ARN
# 2. deploy (takes ~3-6 min)
python deploy.py

# 3. test
python test_endpoint.py
```

Expected output — the high-risk policy should predict noticeably more claims
than the low-risk one:

```
Predicted expected monthly claims:
  policy [55, 3, 9000, 0, 780, 1, 110, 12]   -> 0.05xx
  policy [21, 12, 20000, 2, 580, 5, 250, 12] -> 1.xx
```

## Tear down when done

```bash
python cleanup.py
```

## Feature order

`[driver_age, vehicle_age, annual_mileage, prior_claims, credit_score,
region_risk, vehicle_power, exposure_months]`

## Notes for your portfolio write-up

- This gives SureGuard a **second deployment path** alongside the serverless
  Lambda/Step Functions stack — you can now compare a managed ML endpoint vs a
  hand-rolled serverless API on cost, latency, and operational overhead.
- Uses AWS's **prebuilt sklearn container** + a `requirements.txt`, so no custom
  Docker image is needed. Good talking point: "when to use a prebuilt container
  vs. bring-your-own-container."
- Same 4-function handler pattern (`model_fn` / `input_fn` / `predict_fn` /
  `output_fn`) applies to any framework on SageMaker.
