# Sentence Similarity FastAPI Application

This example demonstrates how to use the ONNX sentence similarity model with FastAPI to create a web API for computing sentence embeddings and similarities.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the ONNX model:
   - Visit [redis/langcache-embed-v1](https://huggingface.co/redis/langcache-embed-v1) on Hugging Face
   - Go to the "Files and versions" tab
   - Download the `langcache-embed-v1-model.onnx` file
   - Place it in the parent directory of this example

## Running the Application

Start the FastAPI application:

```bash
python app.py
```

The server will start at `http://localhost:8000`.

## API Documentation

Once the server is running, you can access:
- Interactive API documentation (Swagger UI): `http://localhost:8000/docs`
- Alternative API documentation (ReDoc): `http://localhost:8000/redoc`

## API Endpoints

### POST /compute_similarity/

Computes embeddings and similarity scores for a list of sentences.

Request body:
```json
{
    "sentences": [
        "First sentence",
        "Second sentence",
        "Third sentence"
    ]
}
```

Response:
```json
{
    "embeddings": [
        [...],  // embedding vector for first sentence
        [...],  // embedding vector for second sentence
        [...]   // embedding vector for third sentence
    ],
    "similarities": [
        {
            "sentence1_index": 0,
            "sentence2_index": 1,
            "sentence1": "First sentence",
            "sentence2": "Second sentence",
            "similarity_score": 0.85
        },
        // ... more similarity pairs
    ]
}
```

## Example Usage with cURL

```bash
curl -X POST "http://localhost:8000/compute_similarity/" \
     -H "Content-Type: application/json" \
     -d '{
           "sentences": [
             "Will the value of Indian rupee increase after the ban of 500 and 1000 rupee notes?",
             "What will be the implications of banning 500 and 1000 rupees currency notes on Indian economy?",
             "Are Danish Sait'\''s prank calls fake?"
           ]
         }'
```

## Example Usage with Python requests

```python
import requests

url = "http://localhost:8000/compute_similarity/"
data = {
    "sentences": [
        "Will the value of Indian rupee increase after the ban of 500 and 1000 rupee notes?",
        "What will be the implications of banning 500 and 1000 rupees currency notes on Indian economy?",
        "Are Danish Sait's prank calls fake?"
    ]
}

response = requests.post(url, json=data)
result = response.json()
print(result)
``` 