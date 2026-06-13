import json
import random
import pickle
import numpy as np
import nltk
import google.generativeai as genai

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


lemmatizer = WordNetLemmatizer()

model = load_model('Chatbot/chatbot_model(N).h5')
intents = json.loads(open('Chatbot/intents.json', encoding='utf-8').read())
words = pickle.load(open('Chatbot/words(N).pkl', 'rb'))
classes = pickle.load(open('Chatbot/classes(N).pkl', 'rb'))


genai.configure(api_key="AIzaSyBvbSNd89WOEk1x92uP7012i-m5IuLOHWI")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]

    ERROR_THRESHOLD = 0.80

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return [{"intent": classes[r[0]], "probability": float(r[1])} for r in results]


TRAVEL_KEYWORDS = [
    "visit","trip","travel","tour","cost","price",
    "ticket","entry","guide","package","how much"
]

def is_travel_related(text):
    t = text.lower()
    return any(k in t for k in TRAVEL_KEYWORDS)


def get_dataset_response(ints):
    if not ints:
        return None

    tag = ints[0]["intent"]

    for i in intents["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])

    return None

# =========================
# GEMINI FALLBACK
# =========================
def ask_gemini(user_text):
    prompt = f"""
You are a helpful India travel assistant.
Answer clearly and correctly.
Do not use markdown or special symbols.
User question: {user_text}
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# =========================
# FINAL CHATBOT RESPONSE
# =========================
def chatbot_response(msg):

    # 1️⃣ Predict intent
    ints = predict_class(msg)

    # 2️⃣ Use dataset only if:
    # high confidence + travel related
    if ints and ints[0]["probability"] > 0.80 and is_travel_related(msg):
        dataset_reply = get_dataset_response(ints)
        if dataset_reply:
            return dataset_reply

    # 3️⃣ Otherwise Gemini
    gemini_reply = ask_gemini(msg)
    return gemini_reply
