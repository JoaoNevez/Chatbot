import json
import nltk
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

nltk.download('punkt')
stemmer = PorterStemmer()

with open("intents.json", "r", encoding='utf-8') as file:
    data = json.load(file)

corpus = []
tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern.lower())
        stemmed = [stemmer.stem(w) for w in tokens]
        corpus.append(" ".join(stemmed))
        tags.append(intent['tag'])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
y = np.array(tags)

model = MultinomialNB()
model.fit(X, y)

# Salvar o modelo e vetorizador
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
