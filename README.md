# Next Word Predictor

A sophisticated AI-powered predictive text application that enhances typing efficiency using a combination of LSTM and BERT models. The application provides intelligent word suggestions based on context, user preferences, and common phrases.

## Features

- **Hybrid Prediction System**: Combines LSTM and BERT models for accurate word predictions
- **Context-Aware Suggestions**: Provides relevant suggestions based on the input context
- **User Preference Learning**: Adapts to user's writing style and preferences
- **Common Phrase Detection**: Includes domain-specific terminology and common phrases
- **RESTful API**: Easy integration with any application
- **Real-time Predictions**: Fast and efficient word suggestions

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Next_Word_Predictor.git
cd Next_Word_Predictor
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:

```bash
uvicorn main:app --reload
```

2. Access the API documentation:

- Open your browser and navigate to `http://localhost:8000/docs`
- Interactive API documentation is available at `http://localhost:8000/redoc`

3. Make predictions:

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Your input text here"}
)
predictions = response.text
```

## API Endpoints

### POST /predict

Predicts the next word based on the input text.

**Request Body:**

```json
{
  "text": "Your input text here"
}
```

**Response:**

```
word1, word2, word3
```

### GET /admin/stats

Retrieves usage statistics and user preferences.

**Response:**

```json
{
    "user_preferences": {
        "word1": count1,
        "word2": count2
    },
    "prediction_usage": {
        "word1": percentage1,
        "word2": percentage2
    }
}
```

## Project Structure

```
Next_Word_Predictor/
├── main.py              # FastAPI application
├── requirements.txt     # Project dependencies
├── lstm_next_word_model.h5  # Trained LSTM model
├── tokenizer.pkl       # Text tokenizer
├── medium_data.csv     # Training data
└── README.md           # Project documentation
```

## Model Architecture

The application uses a hybrid approach combining:

- LSTM model for sequence-based predictions
- BERT model for contextual understanding
- Common phrase dictionary for domain-specific suggestions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- TensorFlow team for the LSTM implementation
- Hugging Face for the BERT model
- FastAPI for the web framework
