# Bring in lightweight dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import json



app = FastAPI()

class LanguageItem(BaseModel):
    text:  str #"ciao", // the test for language detection

with open('trained_pipeline-0.1.0.pk1', 'rb') as f:
    model = pickle.load(f)

with open('language_labels.json', 'r') as f:
    language_labels = json.load(f)

@app.post("/")
async def detection_endpoint(item: LanguageItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)
    predicted_label = yhat[0]
    predicted_language = language_labels[predicted_label]
    return {"predicted language": predicted_language}

    



