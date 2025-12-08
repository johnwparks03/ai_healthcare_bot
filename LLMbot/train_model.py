import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import psycopg2

# Configuration
MODEL_NAME = "medalpaca/medalpaca-7b"
# CSV_PATH = "HealthData/DiseaseClassificationKaggle/questions_and_answers.csv"
OUTPUT_DIR = "./fine_tuned_medalpaca"
LORA_OUTPUT_DIR = "./medalpaca_lora_adapter"
RESULTS_FILE = "./test_results.txt"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load and prepare the dataset
# print("Loading dataset from CSV...")
# df = pd.read_csv(CSV_PATH)
# print(f"Loaded {len(df)} question-answer pairs")


conn = None
cur = None
try:
    conn = psycopg2.connect(
        host="database-1.chsyuesiimuq.us-east-2.rds.amazonaws.com",
        database="postgres",
        user="postgres",
        password="abc12345",
        port="5432"
    )

    cur = conn.cursor()

    train_amt = 10000
    cur.execute(
        'SELECT * FROM public."QuestionAnswer" ORDER BY RANDOM() LIMIT %s',
        (train_amt,) 
    )
    QA = cur.fetchall()

    # ---- Convert to DataFrame ----
    colnames = [desc[0] for desc in cur.description]   # get column names from DB cursor
    df = pd.DataFrame(QA, columns=colnames)
    # print("\nDataFrame Preview:")
    # print(df.head(), "\n")

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


# Format the data for instruction tuning
def format_instruction(row):
    """Format each Q&A pair as an instruction-following prompt."""
    instruction = f"Symptoms: {row['question']}\n Diagnosis: {row['answer']}"
    return instruction

# Create formatted texts
df['text'] = df.apply(format_instruction, axis=1)

# Convert to HuggingFace Dataset - KEEP question and answer columns for evaluation
dataset = Dataset.from_pandas(df[['text', 'question', 'answer']])

# Split into train, validation, and test sets (75-15-10 split)
# First: separate test set (10%)
train_val_test_split = dataset.train_test_split(test_size=0.30, seed=42)
test_dataset = train_val_test_split['test']
train_val_dataset = train_val_test_split['train']

# Second: split remaining 90% into train (75%) and validation (15%)
# 15% of total = 15/90 of remaining ≈ 0.1667
train_val_split = train_val_dataset.train_test_split(test_size=0.1667, seed=42)
train_dataset = train_val_split['train']
eval_dataset = train_val_split['test']

print(f"Training samples:   {len(train_dataset)} ({len(train_dataset) / len(dataset) * 100:.1f}%)")
print(f"Validation samples: {len(eval_dataset)} ({len(eval_dataset) / len(dataset) * 100:.1f}%)")
print(f"Test samples:       {len(test_dataset)} ({len(test_dataset) / len(dataset) * 100:.1f}%)")

# Step 2: Configure quantization for memory efficiency
print("\nConfiguring model for 4-bit quantization (QLoRA)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Step 3: Load the model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# Step 4: Configure LoRA
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Step 5: Configure training arguments
print("\nConfiguring training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    save_total_limit=3,
    remove_unused_columns=False,  # Keep extra columns for evaluation
)

# Step 6: Initialize the trainer
print("Initializing trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    args=training_args
)

# Step 7: Train the model
print("\n" + "=" * 50)
print("Starting training...")
print("=" * 50 + "\n")

trainer.train()

# Step 8: Save the fine-tuned model
print("\n" + "=" * 50)
print("Training complete! Saving model...")
print("=" * 50)

model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)

print(f"\nLoRA adapter saved to: {LORA_OUTPUT_DIR}")
print(f"Training outputs saved to: {OUTPUT_DIR}")


# ============================================================================
# TRAINING CURVES PLOT
# ============================================================================

def plot_training_curves(trainer, save_path="training_curves.png"):
    """
    Plot training and validation loss curves from trainer log history.

    Args:
        trainer: The HuggingFace Trainer object after training
        save_path: Where to save the plot
    """
    log_history = trainer.state.log_history

    # Extract training loss (logged at each logging_steps)
    train_steps = []
    train_losses = []

    # Extract validation loss (logged at each eval_steps)
    eval_steps = []
    eval_losses = []

    for entry in log_history:
        if 'loss' in entry:  # Training loss
            train_steps.append(entry['step'])
            train_losses.append(entry['loss'])
        if 'eval_loss' in entry:  # Validation loss
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training and Validation Loss together
    ax1 = axes[0]
    ax1.plot(train_steps, train_losses, 'b-', label='Training Loss', alpha=0.7)
    ax1.plot(eval_steps, eval_losses, 'r-o', label='Validation Loss', markersize=4)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Perplexity
    ax2 = axes[1]
    eval_perplexities = [np.exp(loss) for loss in eval_losses]
    ax2.plot(eval_steps, eval_perplexities, 'g-o', markersize=4)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Validation Perplexity')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nTraining curves saved to: {save_path}")

    # Also print a summary table
    print("\n" + "=" * 60)
    print("TRAINING LOG SUMMARY")
    print("=" * 60)
    print(f"{'Step':<10} {'Train Loss':<15} {'Val Loss':<15} {'Val PPL':<15}")
    print("-" * 60)

    # Match up train and eval by step (eval happens less frequently)
    for i, step in enumerate(eval_steps):
        # Find closest training loss
        train_idx = min(range(len(train_steps)), key=lambda j: abs(train_steps[j] - step))
        train_loss = train_losses[train_idx] if train_idx < len(train_losses) else float('nan')
        eval_loss = eval_losses[i]
        eval_ppl = np.exp(eval_loss)
        print(f"{step:<10} {train_loss:<15.4f} {eval_loss:<15.4f} {eval_ppl:<15.4f}")

    print("=" * 60)

    return train_steps, train_losses, eval_steps, eval_losses


# Plot the training curves
print("\n" + "=" * 50)
print("Generating training curves...")
print("=" * 50)

train_steps, train_losses, eval_steps, eval_losses = plot_training_curves(
    trainer,
    save_path=os.path.join(OUTPUT_DIR, "training_curves.png")
)


# ============================================================================
# GENERATION-BASED ACCURACY EVALUATION
# ============================================================================

def generate_diagnosis(question, max_new_tokens=128):
    """Generate a diagnosis using the fine-tuned model."""
    prompt = f"Symptoms: {question}\n\nDiagnosis:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            min_new_tokens=5,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the new tokens (not the prompt)
    new_tokens = outputs[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_answer(model_output, true_label):
    """Check if true diagnosis appears in model output (case-insensitive)."""
    return 1 if true_label.lower() in model_output.lower() else 0


def run_accuracy_evaluation(dataset, dataset_name="Test", num_samples=None, verbose=True, save_to_file=None):
    """
    Run generation-based accuracy evaluation on a dataset.

    Args:
        dataset: HuggingFace dataset with 'question' and 'answer' columns
        dataset_name: Name for logging
        num_samples: Number of samples to evaluate (None = all)
        verbose: Print individual predictions
        save_to_file: Path to save detailed results (None = don't save)

    Returns:
        accuracy: float between 0 and 1
        scores: list of individual scores (0 or 1)
        results: list of dicts with detailed results
    """
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))

    print(f"\n{'=' * 80}")
    print(f"EVALUATING ON {dataset_name.upper()} SET ({num_samples} samples)")
    print(f"{'=' * 80}")

    scores = []
    results = []  # Store detailed results

    for i in range(num_samples):
        question = dataset[i]['question']
        true_answer = dataset[i]['answer']

        if verbose:
            print(f"\n[{i + 1}/{num_samples}] Processing...")
            print(f"  Question: {question[:100]}..." if len(question) > 100 else f"  Question: {question}")

        # Generate prediction
        predicted = generate_diagnosis(question)

        if verbose:
            print(f"  Predicted: {predicted[:100]}..." if len(predicted) > 100 else f"  Predicted: {predicted}")
            print(f"  True Answer: {true_answer}")

        # Score: 1 if true answer appears in prediction, 0 otherwise
        score = evaluate_answer(predicted, true_answer)
        scores.append(score)

        # Store detailed result
        results.append({
            'index': i + 1,
            'question': question,
            'true_answer': true_answer,
            'predicted': predicted,
            'correct': score == 1
        })

        if verbose:
            print(f"  Match: {'✓ CORRECT' if score == 1 else '✗ WRONG'}")

    accuracy = np.mean(scores)

    print(f"\n{'=' * 80}")
    print(f"{dataset_name.upper()} SET RESULTS:")
    print(f"  Accuracy: {accuracy:.4f} ({sum(scores)}/{num_samples} correct)")
    print(f"{'=' * 80}")

    # Save results to file if path provided
    if save_to_file:
        save_results_to_file(results, accuracy, dataset_name, save_to_file)

    return accuracy, scores, results


def save_results_to_file(results, accuracy, dataset_name, filepath):
    """Save detailed evaluation results to a text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"EVALUATION RESULTS - {dataset_name.upper()} SET\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"SUMMARY\n")
        f.write(f"-" * 40 + "\n")
        f.write(f"Total Questions: {len(results)}\n")
        f.write(f"Correct: {sum(1 for r in results if r['correct'])}\n")
        f.write(f"Wrong: {sum(1 for r in results if not r['correct'])}\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")

        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")

        # Correct predictions
        f.write("✓ CORRECT PREDICTIONS\n")
        f.write("-" * 40 + "\n\n")
        correct_results = [r for r in results if r['correct']]
        for r in correct_results:
            f.write(f"[{r['index']}] ✓ CORRECT\n")
            f.write(f"  Question: {r['question']}\n")
            f.write(f"  True Answer: {r['true_answer']}\n")
            f.write(f"  Predicted: {r['predicted']}\n")
            f.write("\n")

        f.write("\n")

        # Wrong predictions
        f.write("✗ WRONG PREDICTIONS\n")
        f.write("-" * 40 + "\n\n")
        wrong_results = [r for r in results if not r['correct']]
        for r in wrong_results:
            f.write(f"[{r['index']}] ✗ WRONG\n")
            f.write(f"  Question: {r['question']}\n")
            f.write(f"  True Answer: {r['true_answer']}\n")
            f.write(f"  Predicted: {r['predicted']}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"\nDetailed results saved to: {filepath}")


def plot_test_results(results, save_path="test_results_plot.png"):
    """
    Plot test results showing correct vs wrong predictions.

    Args:
        results: List of result dicts from run_accuracy_evaluation
        save_path: Where to save the plot
    """
    correct = sum(1 for r in results if r['correct'])
    wrong = sum(1 for r in results if not r['correct'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Bar chart
    ax1 = axes[0]
    categories = ['Correct', 'Wrong']
    counts = [correct, wrong]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Count')
    ax1.set_title('Test Set Results')
    ax1.set_ylim(0, max(counts) * 1.2)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Plot 2: Pie chart
    ax2 = axes[1]
    sizes = [correct, wrong]
    labels = [f'Correct\n({correct})', f'Wrong\n({wrong})']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)
    wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.set_title('Accuracy Breakdown')

    # Plot 3: Running accuracy (cumulative)
    ax3 = axes[2]
    cumulative_correct = np.cumsum([1 if r['correct'] else 0 for r in results])
    cumulative_total = np.arange(1, len(results) + 1)
    running_accuracy = cumulative_correct / cumulative_total

    ax3.plot(cumulative_total, running_accuracy, 'b-', linewidth=2)
    ax3.axhline(y=accuracy, color='r', linestyle='--', alpha=0.7, label=f'Final: {accuracy:.2%}')
    ax3.set_xlabel('Question Number')
    ax3.set_ylabel('Cumulative Accuracy')
    ax3.set_title('Running Accuracy Over Test Set')
    ax3.set_ylim(0, 1.0)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle(f'Test Set Evaluation: {accuracy:.2%} Accuracy ({correct}/{total})',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nTest results plot saved to: {save_path}")


# Step 9: Get final validation metrics (best model already loaded)
print("\n" + "=" * 50)
print("Final validation metrics (best model)...")
print("=" * 50 + "\n")

val_results = trainer.evaluate()
print(f"Validation Set Results:")
print(f"  Loss: {val_results['eval_loss']:.4f}")
print(f"  Perplexity: {torch.exp(torch.tensor(val_results['eval_loss'])):.4f}")

# Step 10: Run generation-based accuracy evaluation
print("\n" + "=" * 50)
print("Evaluating generation accuracy on test set...")
print("=" * 50 + "\n")

# Evaluate on full test set with results saved to file
test_accuracy, test_scores, test_detailed_results = run_accuracy_evaluation(
    test_dataset,
    dataset_name="Test",
    num_samples=None,  # All samples
    verbose=True,
    save_to_file=RESULTS_FILE  # Save detailed results to txt
)

# Also evaluate on validation set for comparison (no file save)
val_accuracy, val_scores, _ = run_accuracy_evaluation(
    eval_dataset,
    dataset_name="Validation",
    num_samples=50,  # Just a sample for comparison
    verbose=False,
    save_to_file=None
)

# Final summary
print("\n" + "=" * 80)
print("FINAL EVALUATION SUMMARY")
print("=" * 80)
print(f"  Val Loss (best): {val_results['eval_loss']:.4f}")
print(f"  Val Perplexity:  {torch.exp(torch.tensor(val_results['eval_loss'])):.4f}")
print(f"  Test Accuracy:   {test_accuracy:.4f} ({sum(test_scores)}/{len(test_scores)} correct)")
print(f"  Val Accuracy:    {val_accuracy:.4f} (sample of 50)")
print("=" * 80)

# Plot test results
print("\n" + "=" * 50)
print("Generating test results plot...")
print("=" * 50)

plot_test_results(
    test_detailed_results,
    save_path=os.path.join(OUTPUT_DIR, "test_results_plot.png")
)


# Plot accuracy comparison (original)
def plot_accuracy_results(test_scores, val_scores, save_path="accuracy_results.png"):
    """Plot accuracy distribution and comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Accuracy bar chart (Test vs Val)
    ax1 = axes[0]
    datasets = ['Test Set', 'Validation\n(sample)']
    accuracies = [np.mean(test_scores), np.mean(val_scores)]
    colors = ['#2ecc71', '#3498db']
    bars = ax1.bar(datasets, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy by Dataset')
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    ax1.legend()

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Pie chart of correct vs incorrect
    ax2 = axes[1]
    correct = sum(test_scores)
    incorrect = len(test_scores) - correct
    sizes = [correct, incorrect]
    labels = [f'Correct\n({correct})', f'Incorrect\n({incorrect})']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax2.set_title('Test Set Results Breakdown')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nAccuracy results saved to: {save_path}")


plot_accuracy_results(
    test_scores,
    val_scores,
    save_path=os.path.join(OUTPUT_DIR, "accuracy_results.png")
)

print("\n" + "=" * 50)
print("Training script completed successfully!")
print("=" * 50)
print(f"\nOutputs:")
print(f"  - LoRA adapter: {LORA_OUTPUT_DIR}")
print(f"  - Training curves: {os.path.join(OUTPUT_DIR, 'training_curves.png')}")
print(f"  - Test results plot: {os.path.join(OUTPUT_DIR, 'test_results_plot.png')}")
print(f"  - Detailed results: {RESULTS_FILE}")
print(f"\nTo use the fine-tuned model, load it with:")
print(f"  from peft import PeftModel")
print(f"  base_model = AutoModelForCausalLM.from_pretrained('{MODEL_NAME}')")
print(f"  model = PeftModel.from_pretrained(base_model, '{LORA_OUTPUT_DIR}')")