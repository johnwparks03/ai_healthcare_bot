# -*- coding: utf-8 -*-
"""medalpaca_training"""

import psycopg2
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer


# ---------------------------------------------------------
# LOAD DATA FROM POSTGRES
# ---------------------------------------------------------
def load_data(limit=None):
    try:
        conn = psycopg2.connect(
            host="database-1.chsyuesiimuq.us-east-2.rds.amazonaws.com",
            database="postgres",
            user="postgres",
            password="abc12345",
            port="5432"
        )

        cur = conn.cursor()

        if limit:
            cur.execute('SELECT * FROM public."QuestionAnswer" ORDER BY RANDOM() LIMIT %s', (limit,))
        else:
            cur.execute('SELECT * FROM public."QuestionAnswer"')

        QA = cur.fetchall()

    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        QA = []

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return QA


# ---------------------------------------------------------
# FORMAT DATA FOR INSTRUCTION TUNING
# ---------------------------------------------------------
def format_row(example):
    example["text"] = f"""### Instruction:
{example['question']}

### Response:
{example['answer']}"""
    return example


# ---------------------------------------------------------
# LOAD RAW DATA
# ---------------------------------------------------------
raw_data = load_data(limit=100)

questions = []
answers = []

for q, a in raw_data:
    questions.append(q)
    answers.append(a)

hf_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers
})

hf_dataset = hf_dataset.map(format_row)


# ---------------------------------------------------------
# TRAINING ARGS
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir="medalpaca-trained",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    fp16=True,
    save_strategy="epoch",
    logging_steps=20,
)


# ---------------------------------------------------------
# SFT TRAINER — NO TOKENIZER
# TRL will load model + tokenizer internally
# ---------------------------------------------------------
model_name = "medalpaca/medalpaca-7b"

trainer = SFTTrainer(
    model=model_name,                 # TRL loads tokenizer internally
    args=training_args,
    train_dataset=hf_dataset,
    formatting_func=lambda x: x["text"],
)


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------
trainer.train()


# ---------------------------------------------------------
# SAVE MODEL + TOKENIZER (automatically available)
# ---------------------------------------------------------
trainer.model.save_pretrained("medalpaca-disease-classification")
trainer.tokenizer.save_pretrained("medalpaca-disease-classification")
