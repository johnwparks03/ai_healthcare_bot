from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from talk_to_ai import model_predict


class InputData(BaseModel):
    user_input: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
def  predict(data: InputData):
    
    model_output = model_predict(data.user_input)
    
    return {
        "user_input": data.user_input,
        "model_output": model_output
    }
