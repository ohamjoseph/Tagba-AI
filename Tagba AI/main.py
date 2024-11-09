from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from .llama import generate_prompte


app = FastAPI()

class Query(BaseModel):
    text: str
    lang: str = "Tagba"


@app.get("/")
def read_root():
    return "Tagba AI v0.0.1"


@app.post("/translate/")
def read_item(query: Query):
    response = generate_prompte(inputs=query.text, langue=query.lang )
    return {"translation": response}