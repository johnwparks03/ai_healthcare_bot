from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class InputData(BaseModel):
    text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def  predict(data: InputData):

    #Parse input data and do any pre-processing
    
    return f"User Input: {data.text}\nDiagnosis: You have a cold."

@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/")
def read_root():
    return {"message": "Hello World"}
