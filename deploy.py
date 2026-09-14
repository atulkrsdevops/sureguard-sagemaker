"""
SureGuard -> SageMaker Serverless Inference deployment.

Run this on your machine (with AWS credentials configured for us-east-2).
It will:
  1. upload model.tar.gz to your default SageMaker S3 bucket
  2. create a SageMaker Model using AWS's prebuilt sklearn container
  3. create a SERVERLESS endpoint config (no hourly cost, scales to zero)
  4. deploy the endpoint and wait until it's ready

Prereqs:
  pip install sagemaker boto3
  aws configure   (or env vars)  -> region us-east-2, an IAM user/role with SageMaker access

Notes:
  - Serverless endpoints bill only per request + per-ms of compute, nothing while idle.
  - The IAM *execution role* (ROLE_ARN below) is what SageMaker itself assumes to read S3.
    If you don't have one, see create_role note at the bottom.
"""
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serverless import ServerlessInferenceConfig

# ---------------------------------------------------------------------------
# CONFIG - edit these two if needed
# ---------------------------------------------------------------------------
REGION = "us-east-2"
# Paste your SageMaker execution role ARN here. If you created a SageMaker
# domain/Studio, one already exists (looks like:
#   arn:aws:iam::<account-id>:role/service-role/AmazonSageMaker-ExecutionRole-XXXX)
ROLE_ARN = "arn:aws:iam::332422487493:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
# ---------------------------------------------------------------------------

session = sagemaker.Session(boto_session=None)  # uses default region from aws config
bucket = session.default_bucket()
prefix = "sureguard"

print(f"Region: {session.boto_region_name}")
print(f"S3 bucket: {bucket}")

# 1. Upload the model artifact to S3
model_s3_uri = session.upload_data(
    "model.tar.gz", bucket=bucket, key_prefix=f"{prefix}/model"
)
print(f"Uploaded model to: {model_s3_uri}")

# 2. Define the SageMaker model (prebuilt sklearn container runs our inference.py)
model = SKLearnModel(
    model_data=model_s3_uri,
    role=ROLE_ARN,
    entry_point="inference.py",
    source_dir="code",             # contains inference.py + requirements.txt
    framework_version="1.2-1",     # sklearn container version (CPU)
    sagemaker_session=session,
    name="sureguard-lgbm-model",
)

# 3. Serverless config: 2 GB memory is plenty for a LightGBM model
serverless_cfg = ServerlessInferenceConfig(
    memory_size_in_mb=2048,
    max_concurrency=5,
)

# 4. Deploy
print("\nDeploying serverless endpoint (this takes ~3-6 minutes)...")
predictor = model.deploy(
    serverless_inference_config=serverless_cfg,
    endpoint_name="sureguard-endpoint",
)
print(f"\n✅ Endpoint live: {predictor.endpoint_name}")
print("Test it with:  python test_endpoint.py")
print("Delete it with: python cleanup.py   (do this when done to avoid charges)")
