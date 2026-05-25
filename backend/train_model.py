import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")


fake["label"] = 0
real["label"] = 1


data = pd.concat([fake, real])


data = data.sample(frac=1)


def clean_text(text):

    text = text.lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>+', '', text)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'\n', '', text)

    text = re.sub(r'\w*\d\w*', '', text)

    return text

# Apply cleaning
data["text"] = data["text"].apply(clean_text)


x = data["text"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


vectorizer = TfidfVectorizer(stop_words="english")

xv_train = vectorizer.fit_transform(x_train)

xv_test = vectorizer.transform(x_test)



model = SGDClassifier(
    loss="hinge",
    penalty=None,
    learning_rate="optimal",
    max_iter=1000
)

model.fit(xv_train, y_train)


y_pred = model.predict(xv_test)



score = accuracy_score(y_test, y_pred)

print("Accuracy:", score)


joblib.dump(model, "model.pkl")

joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully")
