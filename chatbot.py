import nltk
import pickle
import json
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Carregar modelo e vetorizador
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Carregar intents
with open("intents.json", "r", encoding='utf-8') as file:
    intents = json.load(file)

def get_response(msg):
    tokens = nltk.word_tokenize(msg.lower())
    stemmed = [stemmer.stem(w) for w in tokens]
    final_input = " ".join(stemmed)
    X_test = vectorizer.transform([final_input])
    tag = model.predict(X_test)[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Desculpe, não entendi."

if __name__ == "__main__":
    print("Chatbot iniciado! (Digite 'sair' para encerrar)")
    while True:
        inp = input("Você: ")
        if inp.lower() == "sair":
            break
        response = get_response(inp)
        print("Bot:", response)
