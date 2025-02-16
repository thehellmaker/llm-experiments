import json
import boto3
import os

# AWS Region and Lambda function name
AWS_REGION = "us-west-2"
TARGET_LAMBDA = os.getenv("TARGET_LAMBDA", "vllm-reverse-proxy")

# Initialize AWS Lambda client
lambda_client = boto3.client("lambda", region_name=AWS_REGION)

def lambda_handler(event, context):
    try:
        # Ensure event contains the necessary data
        if not isinstance(event, dict):
            raise ValueError("Invalid event format. Expected a JSON object.")

        # Call the target Lambda function
        response = lambda_client.invoke(
            FunctionName=TARGET_LAMBDA,
            InvocationType="RequestResponse",  # Synchronous call
            Payload=json.dumps(event)
        )

        # Read the response
        response_payload = json.loads(response["Payload"].read().decode("utf-8"))

        return {
            "statusCode": response.get("StatusCode", 500),
            "body": response_payload
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
