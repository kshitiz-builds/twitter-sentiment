# MoodMosaic — Twitter Sentiment Analysis

Binary sentiment classifier (Positive / Negative) trained on 1.6 million tweets using TF-IDF + Logistic Regression.

**Team T18 | GD Goenka University**

---

## How it works

1. Raw tweet text is cleaned — URLs removed, lowercased, stop words filtered, lemmatized
2. Cleaned text is vectorized with TF-IDF (top 50k features)
3. Logistic Regression predicts Positive / Negative with a confidence score

---

## Setup

```bash
git clone https://github.com/<your-username>/moodmosaic.git
cd moodmosaic
pip install -r requirements.txt
```

---

## Dataset

[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) — 1.6M tweets, labels: `0` = Negative, `4` = Positive (remapped to `1`)

Download and place `training.1600000.processed.noemoticon.csv` in the project root.

---

## Train

```bash
python sentiment_analysis.py training.1600000.processed.noemoticon.csv
```

Saves `trained_model.sav` and `vectorizer.sav` in the project root.

---

## Predict (interactive CLI)

```bash
python sentiment_analysis.py
```

```
Tweet: I love how this turned out
  → Positive  (89.3% confidence)

Tweet: worst experience ever, never going back
  → Negative  (94.1% confidence)
```

---

## Use as a module

```python
from sentiment_analysis import load_model, predict

model, vectorizer = load_model()
result = predict("This is amazing!", model, vectorizer)
print(result)
# {'label': 'Positive', 'confidence': 91.4, 'text': 'This is amazing!'}
```

---

## Requirements

```
pandas
numpy
scikit-learn
nltk
```

---

## Results

| Split | Accuracy |
|-------|----------|
| Train | ~82%     |
| Test  | ~78%     |

---

## Project Structure

```
moodmosaic/
├── sentiment_analysis.py   
├── trained_model.sav       
├── vectorizer.sav          
├── requirements.txt
└── README.md
```
