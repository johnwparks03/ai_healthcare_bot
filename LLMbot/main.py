import os; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import torch
import numpy as np

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load a medical Q&A dataset
dataset = load_dataset("medalpaca/medical_meadow_mediqa", split="train[:100]")

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

# Load model with 8-bit quantization
tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    "medalpaca/medalpaca-7b",
    quantization_config=quantization_config,
    device_map="auto"
)

# Step 2: Simple embedding-based retrieval
def get_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts)


# Create knowledge base
knowledge_base = [item['input'] + " " + item['output'] for item in dataset]
kb_embeddings = get_embeddings(knowledge_base)


# Step 3: Retrieval function
def retrieve_context(query, top_k=3):
    query_embedding = get_embeddings([query])[0]

    # Calculate similarity
    similarities = np.dot(kb_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [knowledge_base[i] for i in top_indices]


# Step 4: LLM for generation using medalpaca model
def generate_answer(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = inputs.to(device)  # Move inputs to GPU

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Step 5: RAG pipeline
def medical_qa(question):
    # Retrieve relevant context
    context = retrieve_context(question)

    # Build prompt with context
    prompt = f"Context: {' '.join(context)}\n\nQuestion: {question}\n\nAnswer:"

    # Generate answer using medalpaca model
    response = generate_answer(prompt, max_length=1024)
    return response


# Test it
question = "What are the symptoms of diabetes?"
answer = medical_qa(question)
print(answer)