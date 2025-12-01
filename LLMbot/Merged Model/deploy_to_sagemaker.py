"""
Deploy using boto3 directly - better error messages than sagemaker SDK
"""
import boto3
import time

# Configuration
MODEL_NAME = "RedoneMedBot-model"
ENDPOINT_CONFIG_NAME = "RedoneMedBot-config"
ENDPOINT_NAME = "RedoneMedBot"
S3_MODEL_PATH = "s3://amazon-sagemaker-313078327074-us-east-2-ayialjl56dkvjd/ModelHouse/model.tar.gz"
ROLE = "arn:aws:iam::313078327074:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
REGION = "us-east-2"
INSTANCE_TYPE = "ml.g4dn.xlarge"

# HuggingFace container image for us-east-2
# From: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
IMAGE_URI = "763104351884.dkr.ecr.us-east-2.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu118-ubuntu20.04"

sagemaker = boto3.client("sagemaker", region_name=REGION)

def create_model():
    print(f"Creating model: {MODEL_NAME}")
    try:
        response = sagemaker.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer={
                "Image": IMAGE_URI,
                "ModelDataUrl": S3_MODEL_PATH,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": S3_MODEL_PATH,
                }
            },
            ExecutionRoleArn=ROLE,
        )
        print(f"Model created: {response['ModelArn']}")
    except sagemaker.exceptions.ClientError as e:
        if "Cannot create already existing model" in str(e):
            print("Model already exists, continuing...")
        else:
            raise e

def create_endpoint_config():
    print(f"Creating endpoint config: {ENDPOINT_CONFIG_NAME}")
    try:
        response = sagemaker.create_endpoint_config(
            EndpointConfigName=ENDPOINT_CONFIG_NAME,
            ProductionVariants=[
                {
                    "VariantName": "primary",
                    "ModelName": MODEL_NAME,
                    "InstanceType": INSTANCE_TYPE,
                    "InitialInstanceCount": 1,
                }
            ],
        )
        print(f"Endpoint config created: {response['EndpointConfigArn']}")
    except sagemaker.exceptions.ClientError as e:
        if "Cannot create already existing endpoint configuration" in str(e):
            print("Endpoint config already exists, continuing...")
        else:
            raise e

def create_endpoint():
    print(f"Creating endpoint: {ENDPOINT_NAME}")
    try:
        response = sagemaker.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=ENDPOINT_CONFIG_NAME,
        )
        print(f"Endpoint creation started: {response['EndpointArn']}")
    except sagemaker.exceptions.ClientError as e:
        if "Cannot create already existing endpoint" in str(e):
            print("Endpoint already exists")
            return
        else:
            raise e

    # Wait for endpoint to be ready
    print("Waiting for endpoint to be InService (this takes 10-15 minutes)...")
    while True:
        response = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = response["EndpointStatus"]
        print(f"  Status: {status}")

        if status == "InService":
            print("Endpoint is ready!")
            break
        elif status == "Failed":
            print(f"Endpoint failed: {response.get('FailureReason', 'Unknown')}")
            break

        time.sleep(30)

def cleanup():
    """Delete everything if needed"""
    print("Cleaning up...")
    try:
        sagemaker.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print("Deleted endpoint")
    except: pass
    try:
        sagemaker.delete_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)
        print("Deleted endpoint config")
    except: pass
    try:
        sagemaker.delete_model(ModelName=MODEL_NAME)
        print("Deleted model")
    except: pass

if __name__ == "__main__":
    # Uncomment to clean up first if needed:
    # cleanup()

    create_model()
    create_endpoint_config()
    create_endpoint()