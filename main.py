from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline
import pickle
from collections import Counter

app = FastAPI()

# Load LSTM model
model = tf.keras.models.load_model("lstm_next_word_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load BERT model
bert_predictor = pipeline("fill-mask", model="bert-base-uncased")

# Common phrases dictionary
common_phrases = {
    # Technology and AI
    "machine": ["learning", "intelligence", "vision", "translation", "perception", "automation", "reasoning", "ethics", "algorithms", "optimization"],
    "data": ["science", "analytics", "engineering", "mining", "visualization", "warehousing", "pipelines", "governance", "cleaning", "storage"],
    "artificial": ["intelligence", "neural networks", "learning", "agents", "reasoning", "creativity", "ethics", "vision", "planning", "decision-making"],
    "deep": ["learning", "neural networks", "vision", "reinforcement", "generative models", "architectures", "frameworks", "optimization", "interpretability", "applications"],
    "neural": ["networks", "models", "layers", "activation", "architecture", "training", "optimization", "pruning", "compression", "interpretability"],
    "computer": ["vision", "science", "graphics", "networks", "hardware", "security", "architecture", "programming", "engineering", "simulation"],
    "ai": ["ethics", "applications", "research", "development", "frameworks", "tools", "algorithms", "trends", "challenges", "future"],
    "algorithm": ["design", "optimization", "analysis", "complexity", "efficiency", "performance", "scalability", "parallelism", "heuristics", "visualization"],
    "software": ["development", "engineering", "architecture", "testing", "maintenance", "design", "deployment", "security", "automation", "quality"],
    "hardware": ["design", "architecture", "optimization", "performance", "security", "scalability", "maintenance", "integration", "testing", "innovation"],
    "cloud": ["computing", "storage", "services", "platforms", "security", "infrastructure", "migration", "management", "automation", "optimization"],
    "cyber": ["security", "attacks", "defense", "threats", "crime", "resilience", "forensics", "intelligence", "awareness", "policy"],
    "blockchain": ["technology", "applications", "security", "networks", "transactions", "smart contracts", "decentralization", "scalability", "governance", "innovation"],
    "iot": ["devices", "applications", "security", "networks", "automation", "analytics", "integration", "platforms", "sensors", "connectivity"],
    "quantum": ["computing", "mechanics", "physics", "algorithms", "entanglement", "cryptography", "simulation", "communication", "sensors", "optimization"],
    "robotics": ["automation", "control", "vision", "navigation", "manipulation", "sensors", "learning", "planning", "collaboration", "ethics"],
    "virtual": ["reality", "environments", "simulation", "training", "applications", "interaction", "design", "immersion", "platforms", "gaming"],
    "augmented": ["reality", "intelligence", "systems", "applications", "design", "interaction", "platforms", "visualization", "gaming", "training"],
    "natural": ["language", "processing", "selection", "resources", "phenomena", "understanding", "generation", "translation", "interaction", "models"],
    "big": ["data", "analytics", "datasets", "challenges", "opportunities", "processing", "storage", "visualization", "mining", "governance"],
    "automation": ["tools", "processes", "testing", "frameworks", "pipelines", "deployment", "monitoring", "optimization", "security", "integration"],
    "devops": ["practices", "tools", "automation", "pipelines", "deployment", "monitoring", "security", "collaboration", "scalability", "integration"],
    "agile": ["methodology", "development", "practices", "frameworks", "teams", "planning", "delivery", "collaboration", "tools", "management"],
    "scrum": ["framework", "teams", "sprints", "planning", "delivery", "collaboration", "tools", "management", "metrics", "improvement"],
    "ui": ["design", "development", "frameworks", "interaction", "prototyping", "testing", "accessibility", "usability", "tools", "trends"],
    "ux": ["design", "research", "testing", "prototyping", "accessibility", "usability", "tools", "trends", "frameworks", "interaction"],
    "api": ["design", "development", "integration", "security", "management", "documentation", "testing", "performance", "scalability", "automation"],
    "microservices": ["architecture", "development", "deployment", "scalability", "security", "integration", "management", "testing", "performance", "automation"],
    "serverless": ["architecture", "development", "deployment", "scalability", "security", "integration", "management", "testing", "performance", "automation"],
    "containerization": ["tools", "platforms", "deployment", "scalability", "security", "integration", "management", "testing", "performance", "automation"],
    "kubernetes": ["clusters", "deployment", "scalability", "security", "integration", "management", "testing", "performance", "automation", "tools"],
    "docker": ["containers", "deployment", "scalability", "security", "integration", "management", "testing", "performance", "automation", "tools"],
}

# Define input model for FastAPI
class TextInput(BaseModel):
    text: str

# Function to predict next word using LSTM
def predict_with_lstm(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    if not token_list:  # If input is not recognized by tokenizer
        return "No prediction"

    token_list = pad_sequences([token_list], maxlen=50-1, padding="pre")
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return "No prediction"

# Function to predict next word using BERT
def predict_with_bert(text):
    masked_text = text + " [MASK]."
    result = bert_predictor(masked_text)
    return [prediction["token_str"] for prediction in result[:3]]

# Function to get top 3 predicted words
def get_best_predictions(common, lstm, bert):
    word_counts = Counter()

    # Add words to counter
    word_counts.update(common)
    word_counts.update(bert)
    
    if lstm != "No prediction":
        word_counts[lstm] += 1  # Boost LSTM's prediction if valid

    # Get the top 3 words
    best_words = [word for word, _ in word_counts.most_common(3)]
    
    return best_words

@app.post("/predict")
async def predict(input_data: TextInput):
    input_text = input_data.text.strip()

    if not input_text:
        return {"error": "Input text cannot be empty"}

    # Check predefined common phrases
    words = input_text.split()
    last_word = words[-1] if words else ""
    phrase_suggestions = common_phrases.get(last_word.lower(), [])

    # Predict using LSTM
    lstm_suggestion = predict_with_lstm(input_text)

    # Predict using BERT
    bert_suggestions = predict_with_bert(input_text)

    # Get best 3 predictions
    best_predictions = get_best_predictions(phrase_suggestions, lstm_suggestion, bert_suggestions)

      # Format the output as a comma-separated string WITHOUT quotes
    best_predictions_str = ', '.join(best_predictions)

    return best_predictions_str
