import re
import pickle
import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "nor"}


def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)


def load_data(csv_path):
    columns = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(csv_path, names=columns, encoding="ISO-8859-1")
    df.replace({"target": {4: 1}}, inplace=True)
    return df


def train(csv_path, model_path="trained_model.sav", vectorizer_path="vectorizer.sav"):
    print("Loading data...")
    df = load_data(csv_path)

    print("Cleaning text...")
    df["cleaned"] = df["text"].apply(clean_text)

    X = df["cleaned"].values
    Y = df["target"].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )

    vectorizer = TfidfVectorizer(max_features=50000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training model...")
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vec, Y_train)

    train_acc = accuracy_score(Y_train, model.predict(X_train_vec))
    test_acc = accuracy_score(Y_test, model.predict(X_test_vec))
    print(f"Train accuracy : {train_acc:.4f}")
    print(f"Test accuracy  : {test_acc:.4f}")
    print(classification_report(Y_test, model.predict(X_test_vec), target_names=["Negative", "Positive"]))

    pickle.dump(model, open(model_path, "wb"))
    pickle.dump(vectorizer, open(vectorizer_path, "wb"))
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")


def load_model(model_path="trained_model.sav", vectorizer_path="vectorizer.sav"):
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    return model, vectorizer


def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    label = "Positive" if proba[1] >= 0.5 else "Negative"
    confidence = max(proba) * 100
    return {"label": label, "confidence": round(confidence, 2), "text": text}


def interactive(model_path="trained_model.sav", vectorizer_path="vectorizer.sav"):
    model, vectorizer = load_model(model_path, vectorizer_path)
    print("\nMoodMosaic — Sentiment Analyzer")
    print("Type a tweet and press Enter. Type 'quit' to exit.\n")
    while True:
        text = input("Tweet: ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        result = predict(text, model, vectorizer)
        print(f"  → {result['label']}  ({result['confidence']:.1f}% confidence)\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        train(sys.argv[1])
    else:
        print("Usage:")
        print("  Train  : python sentiment_analysis.py training.1600000.processed.noemoticon.csv")
        print("  Predict: import and call predict(text, model, vectorizer)")
        print("\nDataset: https://www.kaggle.com/datasets/kazanova/sentiment140\n")
        interactive()
