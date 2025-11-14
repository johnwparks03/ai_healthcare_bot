from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


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

    #Parse input data and do any pre-processing
    
    return {
        "user_input": data.user_input,
        "model_output": "You have a cold"
    }
