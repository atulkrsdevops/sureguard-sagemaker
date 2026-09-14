"""
Tear down the SureGuard endpoint, endpoint-config, and model.
Run this when you're done to make sure nothing lingers.
(Serverless endpoints don't bill while idle, but clean teardown is good hygiene.)
"""
import boto3

REGION = "us-east-2"
ENDPOINT = "sureguard-endpoint"
CONFIG = "sureguard-endpoint"     # deploy() names the config after the endpoint
MODEL = "sureguard-lgbm-model"

sm = boto3.client("sagemaker", region_name=REGION)

for name, fn in [
    (ENDPOINT, sm.delete_endpoint),
    (CONFIG, sm.delete_endpoint_config),
    (MODEL, sm.delete_model),
]:
    try:
        kwarg = (
            {"EndpointName": name} if fn == sm.delete_endpoint
            else {"EndpointConfigName": name} if fn == sm.delete_endpoint_config
            else {"ModelName": name}
        )
        fn(**kwarg)
        print(f"Deleted: {name}")
    except sm.exceptions.ClientError as e:
        print(f"Skip {name}: {e.response['Error']['Message']}")

print("Cleanup done.")
