'''This script:
Gets questions from database, lives on AWS
Feeds them into the LLM
Uses x test to see how close the LLM's (medalpaca-7b or our model) answer is to the actual answer
Prints out the results'''

"""1- Get questions from database, lives on AWS"""
import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import psycopg2
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import random_color_hex as RCH; RCH.JupyterReset()
import matplotlib.pyplot as plt

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

conn = None
cur = None

AmtOfQuestions = int(input("How many questions would you like to get (NOTE: Default is 100, so probably that)? "))

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

    # print()
    # for i, row in enumerate(QA, 1):
    #     print(f"Q{i}: {row[0]}")
    #     print(f"A{i}: {row[1]}\n")
    # print(f"All fetched questions and answers:\n{QA}")

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

quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained("medalpaca/medalpaca-7b",quantization_config=quantization_config,device_map="auto")

# Go ask the AI the question
def GoAskAI(prompt, max_length=996) -> str:
    newprompt = f"Symptoms: {prompt}\n\nDiagnosis:"

    inputs = tokenizer(newprompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            min_new_tokens=5,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# Evaluation function
def evaluate_answer(model_output, true_label):
    return 1 if true_label.lower() in model_output.lower() else 0

# Test on multiple questions
def test_rag_system(num_samples=10):
    print("\n" + "=" * 80)
    print("EVALUATING RAG SYSTEM")
    print("=" * 80)

    scores = []

    for i, item in enumerate(QA[:num_samples]):
        question = item[0]
        yi = item[1]

        # print(f"\n[{i + 1}/{num_samples}] Processing...")
        # print(f"  Question: {question}")

        yhat = GoAskAI(question)

        # print(f"  Model Answer: {yhat}\n")
        # print(f"  Real Answer: {yi}\n")

        score = evaluate_answer(yhat, yi)
        scores.append(score)
        # print(f"  Score: {score:.3f}")

    avg_score = np.mean(scores)
    print(f"\n{'=' * 80}")
    print(f"AVERAGE SCORE: {avg_score:.3f}")
    print(f"{'=' * 80}")
    return scores


TestInputs = [questions for questions, _ in QA]
Yi = [Answers for _, Answers in QA]

# Run evaluation
ModelScores = test_rag_system(num_samples=len(QA))

print(f"Model Scored: {np.mean(ModelScores)}")

# Plot
correct = sum(ModelScores)
incorrect = len(ModelScores) - correct

plt.bar(['Right', 'Wrong'], [correct, incorrect], color=['#108f46','#e74c3c'])
plt.xlabel('Result')
plt.ylabel('Count')
plt.title(f'Model Accuracy: {correct}/{len(ModelScores)} ({100*correct/len(ModelScores):.1f}%)')
plt.tight_layout()
plt.show()