import json
import urllib.request

# Define model-to-endpoint mapping
MODEL_ENDPOINTS = {
    "deepseek-ai/DeepSeek-R1": "http://<vllm_head_node_ip>:8000/v1/completions",
}

def lambda_handler(event, context):
    try:
        # Ensure the event has the correct structure
        if not isinstance(event, dict):
            raise ValueError("Invalid event format. Expected a JSON object.")

        # Extract model name from event payload
        model_name = event.get("model", "").strip()

        # Fail immediately if model is missing
        if not model_name:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Model name is required"})
            }

        # Ensure the model is explicitly allowed
        vllm_api_url = MODEL_ENDPOINTS.get(model_name)
        if not vllm_api_url:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Unknown model '{model_name}'. Allowed models: {list(MODEL_ENDPOINTS.keys())}"})
            }

        # Prepare request payload
        request_data = json.dumps(event).encode("utf-8")

        # Make HTTP request
        req = urllib.request.Request(
            vllm_api_url, 
            data=request_data, 
            headers={"Content-Type": "application/json"}, 
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            response_data = response.read().decode("utf-8")
            return {
                "statusCode": response.getcode(),
                "body": json.loads(response_data)
            }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
