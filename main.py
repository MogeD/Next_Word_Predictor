from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline
from collections import Counter

app = FastAPI()

# Load LSTM model
model = tf.keras.models.load_model("lstm_next_word_model.h5")

# Load tokenizer (should be saved from training)
import pickle
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load BERT model
bert_predictor = pipeline("fill-mask", model="bert-base-uncased")

# Common phrases dictionary
common_phrases = {
    "machine": ["learning", "intelligence", "vision"],
    "data": ["science", "analytics", "engineering"],
    "artificial": ["intelligence", "neural networks", "learning"],
    "deep": ["learning", "neural networks", "vision"]
}

# Function to predict next word using LSTM
def predict_with_lstm(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=50-1, padding="pre")
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

# Function to predict next word using BERT
def predict_with_bert(text):
    masked_text = text + " [MASK]."
    result = bert_predictor(masked_text)
    return [prediction["token_str"] for prediction in result[:3]]

# FastAPI Endpoint
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    input_text = data["text"]

    # Check predefined common phrases
    words = input_text.split()
    last_word = words[-1] if words else ""
    phrase_suggestions = common_phrases.get(last_word, [])

    # Predict using LSTM
    lstm_suggestion = predict_with_lstm(input_text)

    # Predict using BERT
    bert_suggestions = predict_with_bert(input_text)

    return {
        "common_phrases": phrase_suggestions,
        "lstm_suggestion": lstm_suggestion,
        "bert_suggestions": bert_suggestions
    }
