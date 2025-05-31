# Enhanced Sentence Embeddings API

This example demonstrates an enhanced version of the sentence similarity API that provides two separate endpoints:
1. Generating sentence embeddings
2. Computing sentence similarities

## Features

- **Embedding Generation**: Get embeddings for sentences without computing similarities
- **Similarity Computation**: Calculate similarities between all pairs of input sentences
- **Efficient Processing**: Reusable code for embedding generation
- **Clean API Design**: Separate endpoints for different functionalities

## Requirements

- Python 3.10 - 3.11 (Python 3.13 is not yet supported by ONNX Runtime 1.16.3)

## Setup

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment with Python 3.11:
```bash
uv venv --python=python3.11
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate     # On Windows

uv pip install -r requirements.txt
```

3. Make sure you have the ONNX model file (`langcache-embed-v1-model.onnx`) in the parent directory.

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

## Usage

1. Start the server:
```bash
python app.py
```

2. Run the example client:
```bash
python client.py
```

The client demonstrates both:
- Getting embeddings only
- Computing similarities between sentences

## Example Request/Response

### Generate Embeddings

```python
# Request
{
    "sentences": [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn fox leaps above the sleepy hound"
    ]
}

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

```python
# Request
{
    "sentences": [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn fox leaps above the sleepy hound"
    ]
}

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

## Benefits of Separate Endpoints

1. **Flexibility**: Users can choose to only generate embeddings if they don't need similarities
2. **Efficiency**: No unnecessary similarity computations when only embeddings are needed
3. **Storage**: Users can store embeddings for later use or comparison
4. **Integration**: Easier integration with other systems that might only need embeddings 