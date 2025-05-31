# Enhanced Sentence Embeddings API

This example demonstrates how to use the ONNX sentence similarity model with FastAPI to create a web API for computing sentence embeddings and similarities.

## Features

1. Generating sentence embeddings
2. Computing sentence similarities

## Setup

1. Install the required dependencies:
```bash
uv pip install -r requirements.txt
```

2. Download the ONNX model:
   - Visit [redis/langcache-embed-v1](https://huggingface.co/redis/langcache-embed-v1) on Hugging Face
   - Go to the "Files and versions" tab
   - Download the `langcache-embed-v1-model.onnx` file
   - Place it in the parent directory of this example

## Usage

1. Start the server:
```bash
uv run app.py
```
The server will start at `http://localhost:8000`.

2. Run the example client:
```bash
uv run client.py
```

The client demonstrates both:
- Getting embeddings only
- Computing similarities between sentences


## API Endpoints

### 1. Generate Embeddings
- **Endpoint**: `/generate_embeddings/`
- **Method**: POST
- **Input**: List of sentences
- **Output**: Embeddings for each sentence and the original sentences

### 2. Compute Similarities
- **Endpoint**: `/compute_similarity/`
- **Method**: POST
- **Input**: List of sentences
- **Output**: Similarity scores between all pairs of sentences


## Example Request/Response

### Generate Embeddings

#### Using curl:
```bash
curl -X POST "http://localhost:8000/generate_embeddings/" \
     -H "Content-Type: application/json" \
     -d '{
           "sentences": [
             "The quick brown fox jumps over the lazy dog",
             "A fast auburn fox leaps above the sleepy hound"
           ]
         }'
```


```python
# Response
{
    "embeddings": [[...], [...]],  # List of embedding vectors
    "sentences": [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn fox leaps above the sleepy hound"
    ]
}
```

### Compute Similarities

#### Using curl:
```bash
curl -X POST "http://localhost:8000/compute_similarity/" \
     -H "Content-Type: application/json" \
     -d '{
           "sentences": [
             "The quick brown fox jumps over the lazy dog",
             "A fast auburn fox leaps above the sleepy hound",
             "The weather is nice today"
           ]
         }'
```

```python
# Response
{
    "similarities": [
        {
            "sentence1_index": 0,
            "sentence2_index": 1,
            "sentence1": "The quick brown fox jumps over the lazy dog",
            "sentence2": "A fast auburn fox leaps above the sleepy hound",
            "similarity_score": 0.9876
        }
    ]
}
```