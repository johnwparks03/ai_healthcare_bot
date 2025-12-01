import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def model_fn(model_dir, context=None):  # FIXED: Added context parameter
    """Load model from the model_dir"""
    print(f"Loading model from {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return {"model": model, "tokenizer": tokenizer}


def input_fn(request_body, request_content_type):
    """Parse input data"""
    import json
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return data
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_dict):
    """Generate prediction"""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    question = input_data.get("question", input_data.get("text", ""))
    max_length = input_data.get("max_length", 256)
    temperature = input_data.get("temperature", 0.7)

    # Format prompt (same as your training format)
    prompt = f"""Below is a medical question. Write a response that appropriately answers the question.

### Question:
{question}

### Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the answer
    if "### Answer:" in response:
        answer = response.split("### Answer:")[1].strip()
    else:
        answer = response

    return {"answer": answer}


def output_fn(prediction, response_content_type):
    """Format output"""
    import json
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")