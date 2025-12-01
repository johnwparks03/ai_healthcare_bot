import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Configuration
BASE_MODEL_NAME = "medalpaca/medalpaca-7b"
LORA_ADAPTER_PATH = "./medalpaca_lora_adapter"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configure quantization for inference
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load the fine-tuned LoRA adapter
print("Loading fine-tuned LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
print("Model loaded successfully!")

def generate_answer(question, max_length=256):
    """
    Generate an answer using the fine-tuned model.

    Args:
        question (str): The medical question to answer
        max_length (int): Maximum length of the generated answer

    Returns:
        str: The generated answer
    """
    prompt = f"""Below is a medical question. Write a response that appropriately answers the question.

### Question:
{question}

### Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the answer part
    if "### Answer:" in response:
        answer = response.split("### Answer:")[1].strip()
        return answer
    return response


def interactive_qa():
    """Run an interactive Q&A session."""
    print("\n" + "="*60)
    print("Fine-Tuned Medical Q&A Chatbot")
    print("="*60)
    print("Type 'quit' or 'exit' to stop")
    print("-"*60 + "\n")

    while True:
        question = input("Your question: ")

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question.strip():
            continue

        print("\nThinking...\n")
        answer = generate_answer(question)
        print(f"Answer: {answer}\n")
        print("-"*60 + "\n")


if __name__ == "__main__":
    # Test with some example questions
    print("\n" + "="*60)
    print("Testing the fine-tuned model with sample questions")
    print("="*60 + "\n")

    test_questions = [
        "I am experiencing skin rash, dischromic patches, nodal skin eruptions, and itching. What condition does this match?",
        "My symptoms are continuous sneezing, shivering, and chills. What could this be?",
        "I have watering from eyes, shivering, and continuous sneezing. What do I have?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}: {question}")
        answer = generate_answer(question)
        print(f"Answer: {answer}")
        print("-"*60 + "\n")

    # Start interactive session
    interactive_qa()