from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import re
import string



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load("model.pkl")

vectorizer = joblib.load("vectorizer.pkl")


class NewsRequest(BaseModel):
    text: str



def clean_text(text):

    text = text.lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>+', '', text)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'\n', '', text)

    text = re.sub(r'\w*\d\w*', '', text)

    return text



@app.get("/")
def home():

    return {
        "message": "TruthLens AI Backend Running"
    }


@app.post("/predict")
def predict(news: NewsRequest):

    cleaned_text = clean_text(news.text)

    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]

    confidence = model.decision_function(vectorized_text)[0]

    return {
    "prediction": "REAL" if prediction == 1 else "FAKE",
    "confidence": round(abs(confidence) * 10, 2)
}
