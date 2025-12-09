# AI Healthcare Bot

Our Deep Learning project

## Run Locally

```bash
  git clone https://github.com/johnwparks03/ai_healthcare_bot.git
```

Go to the project directory

```bash
  cd ai_healthcare_bot
```

## Training Data

Our human-annotated medical question and answer dataset is stored in `HealthData`.

## Training the Model

We used `LLMbot/train_model.py` to fine-tune the model.

## Fine-Tuned Model Weights

`fine_tuned_medalpaca/checkpoint-1095/adapter_model.safetensors` contains the LoRA weights. You can apply these weights on top of the base model to recreate the fine-tuned model.

`model.tar.gz` contains the complete model after merging the base model and the fine-tuned weights.

## Experiment with the model

### Start the backend

Ensure you have Python and pip installed

Go to the backend directory

```bash
  cd backend
```

Install the required libraries

```bash
  pip install -r requirements.txt
```

Change to the app directory

```bash
  cd app
```

Start the backend server and listen on port 8000

```bash
  uvicorn main:app --reload --port 8000
```

### Start the frontend

Ensure you have Node (https://nodejs.org/en) and npm installed.

Verify the installion using

```bash
node -v
npm -v
```

Install Angular globally (v16 or later recommended)

```bash
npm install -g @angular/cli
```

Change to the Angular project directory from the root project directory

```bash
cd frontend/ai_healthcare_bot
```

Install dependencies

```bash
npm install
```

Start the web server

```bash
ng serve
```

You can now view the website at http://localhost:4200/
