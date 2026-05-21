import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASETS
# =========================

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

# =========================
# ADD LABELS
# =========================

fake["label"] = 0
real["label"] = 1

# 0 = Fake
# 1 = Real

# =========================
# COMBINE DATA
# =========================

data = pd.concat([fake, real])

# Shuffle rows
data = data.sample(frac=1)

# =========================
# TEXT CLEANING FUNCTION
# =========================

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

# =========================
# FEATURES & LABELS
# =========================

x = data["text"]
y = data["label"]

# =========================
# TRAIN TEST SPLIT
# =========================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TF-IDF VECTORIZATION
# =========================

vectorizer = TfidfVectorizer(stop_words="english")

xv_train = vectorizer.fit_transform(x_train)

xv_test = vectorizer.transform(x_test)

# =========================
# TRAIN MODEL
# =========================

model = SGDClassifier(
    loss="hinge",
    penalty=None,
    learning_rate="optimal",
    max_iter=1000
)

model.fit(xv_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(xv_test)

# =========================
# ACCURACY
# =========================

score = accuracy_score(y_test, y_pred)

print("Accuracy:", score)

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "model.pkl")

joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully")