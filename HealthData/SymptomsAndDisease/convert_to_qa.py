import random
from pathlib import Path

import inflect
import pandas as pd

ROOT = Path(__file__).resolve().parent

data = pd.read_csv(f"{ROOT}/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
p = inflect.engine()
data_cols = [c for c in data.columns if c.lower() != "diseases"]

qa_pairs = []

question_templates = [
    "I have these symptoms: {}. What do I have?",
    "My symptoms are {}. What could this be?",
    "I am experiencing {}. What condition does this match?",
]

i = 1
for _, row in data.iterrows():
    print(i)
    i += 1
    symptoms = [col.replace('_', ' ') for col in data_cols if row[col] == 1]
    random.shuffle(symptoms)
    symptom_text = p.join(symptoms)

    question = random.choice(question_templates).format(symptom_text)
    answer = row["diseases"]

    qa_pairs.append({"question": question, "answer": answer})

qa_df = pd.DataFrame(qa_pairs)
qa_df.to_csv("questions_and_answers.csv", index=False)
