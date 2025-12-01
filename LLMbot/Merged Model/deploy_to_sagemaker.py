"""
Deploy your merged model to SageMaker.
Run this AFTER uploading model.tar.gz to S3.
"""
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# Configuration - UPDATE THESE
S3_MODEL_PATH = "s3://amazon-sagemaker-313078327074-us-east-2-ayialjl56dkvjd/ModelHouse/model.tar.gz"
ROLE = "arn:aws:iam::313078327074:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
INSTANCE_TYPE = "ml.g4dn.xlarge"  # GPU instance (needed for 7B model)


def deploy_model():
    # Create HuggingFace Model
    # NOTE: These versions must match available SageMaker HuggingFace containers!
    # See: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    huggingface_model = HuggingFaceModel(
        model_data=S3_MODEL_PATH,
        role=ROLE,
        transformers_version="4.37.0",  # SageMaker supported version
        pytorch_version="2.1.0",         # SageMaker supported version
        py_version="py310",              # SageMaker supported version
    )

    # Deploy to endpoint
    print("Deploying model to SageMaker endpoint...")
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name="medalpaca-endpoint",
    )

    print(f"Endpoint deployed: {predictor.endpoint_name}")
    return predictor


def test_endpoint(predictor):
    """Test the deployed endpoint"""
    response = predictor.predict({
        "question": "What are the symptoms of diabetes?",
        "max_length": 256,
        "temperature": 0.7
    })
    print(f"Response: {response}")


if __name__ == "__main__":
    predictor = deploy_model()
    test_endpoint(predictor)