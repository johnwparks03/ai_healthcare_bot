'''This script:
Gets questions from database, lives on AWS
Feeds them into the LLM
Uses x test to see how close the LLM's (medalpaca-7b or our model) answer is to the actual answer
Prints out the results'''

"""1- Get questions from database, lives on AWS"""

import psycopg2
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re
from numpy import mean
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import random_color_hex as RCH
import matplotlib.pyplot as plt

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

conn = None
cur = None

AmtOfQuestions = int(input("How many questions do you want? ")) #I think we can trust ourselves not to need error handeling here

try:
    conn = psycopg2.connect(
        host="database-1.chsyuesiimuq.us-east-2.rds.amazonaws.com",
        database="postgres",
        user="postgres",
        password="abc12345",
        port="5432"
    )

    cur = conn.cursor()

    cur.execute(
        'SELECT * FROM public."QuestionAnswer" ORDER BY RANDOM() LIMIT %s',
        (AmtOfQuestions,)
    )
    QA = cur.fetchall()

    print()
    for i, row in enumerate(QA, 1):
        print(f"Q{i}: {row[0]}")
        print(f"A{i}: {row[1]}\n")
    print(f"All fetched questions and answers:\n{QA}")

except psycopg2.Error as e:
    print(f"Error connecting to PostgreSQL: {e}")

finally:
    if cur is not None:
        cur.close()
    if conn is not None:
        conn.close()

"""2- Feed questions into LLM"""

'''Since we dont have reasoning with the answers yet, we will tell the model to provide classification only. 
Then, in a seperate line, we will provide the justification. 
We will grade based off how good the classification is, and just hope that it carrys over for justification (at least for now)'''

tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b", use_fast=False)

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True,bnb_8bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("medalpaca/medalpaca-7b",quantization_config=quantization_config,device_map="auto")

#Go ask the AI the question
def GoAskAI(prompt, max_length=512)->str:
    newprompt = f"I want you to classify a disease/problem I have. When you answer, please follow the following format:\nClassification: Your Answer\nJustification: Why you think the answer is correct\n\n" + prompt
    inputs = tokenizer(newprompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    ModelAnswer=tokenizer.decode(outputs[0], skip_special_tokens=True)
    return ModelAnswer

# Evaluation function
def evaluate_answer(yhat, yi):
    """
    Compare predicted answer to reference answer using semantic similarity.
    Extracts classification from both texts before comparison.
    Returns a score from 0 (completely different) to 1 (identical meaning).
    """
    def extract_classification(text):
        """Extract text that comes after 'Classification: ' using regex"""
        match = re.search(r'Classification:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            # If no classification found, return the full text
            return text
        
    def get_embedding(text):
        """Get embedding from model's hidden states"""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use mean of last hidden state as embedding
            embeddings = outputs.hidden_states[-1].mean(dim=1)

        return embeddings.cpu().numpy()

    # Extract classification from both predicted and reference answers
    yhat_classification = extract_classification(yhat)
    yi_classification = extract_classification(yi)

    print(f"\n  Extracted prediction: {yhat_classification[:80]}...\n")
    print(f"\n  Extracted reference: {yi_classification[:80]}...\n")

    yhat_emb = get_embedding(yhat_classification)
    yi_emb = get_embedding(yi_classification)

    similarity = cosine_similarity(yhat_emb, yi_emb)[0][0]
    # Normalize to 0-1 range
    normalized_similarity = (similarity + 1) / 2
    return normalized_similarity

# Test on multiple questions
def test_rag_system(num_samples=10):
    print("\n" + "=" * 80)
    print("EVALUATING RAG SYSTEM")
    print("=" * 80)

    scores = []

    for i, item in enumerate(QA[:num_samples]):
        question = item[0]
        yi = item[1]

        print(f"\n[{i + 1}/{num_samples}] Processing...")
        yhat = GoAskAI(question)

        score = evaluate_answer(yhat, yi)
        scores.append(score)

        print(f"Q: {question[:80]}...")
        print(f"Reference: {yi[:80]}...")
        print(f"Predicted: {yhat[:80]}...")
        print(f"Score: {score:.3f}")

    avg_score = mean(scores)
    print(f"\n{'=' * 80}")
    print(f"AVERAGE SCORE: {avg_score:.3f}")
    print(f"{'=' * 80}")
    return scores

TestInputs=[questions for questions, _ in QA]
Yi=[Answers for _, Answers in QA]

#Send questions to LLM
#Get its answers
#Use semantic similarity to grade the answers, make a plot of ROC curve

# Run evaluation instead of single test
test_rag_system(num_samples=len(QA))