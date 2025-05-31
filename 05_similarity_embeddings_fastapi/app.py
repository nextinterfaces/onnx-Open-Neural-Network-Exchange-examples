from typing import List, Optional
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

app = FastAPI(
    title="Sentence Similarity API",
    description="API for computing sentence embeddings and similarities using ONNX",
    version="1.0.0"
)

class SentencesInput(BaseModel):
    sentences: List[str]

class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]
    sentences: List[str]

class SimilarityResponse(BaseModel):
    similarities: List[dict]

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def mean_pooling(token_embeddings, attention_mask):
    """Perform mean pooling on token embeddings using attention mask."""
    input_mask_expanded = attention_mask[..., np.newaxis]
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(attention_mask.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask[:, np.newaxis]

def generate_embedding(sentence: str, tokenizer, ort_session):
    """Generate embedding for a single sentence."""
    inputs = tokenizer(
        sentence,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )
    
    ort_inputs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }
    
    token_embeddings = ort_session.run(None, ort_inputs)[0]
    sentence_embedding = mean_pooling(token_embeddings, inputs['attention_mask'])
    return sentence_embedding[0]

# Initialize tokenizer and model at startup
tokenizer = None
ort_session = None

@app.on_event("startup")
async def startup_event():
    global tokenizer, ort_session
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('redis/langcache-embed-v1')
    print("Loading ONNX model...")
    ort_session = ort.InferenceSession("../langcache-embed-v1-model.onnx")

@app.post("/generate_embeddings/", response_model=EmbeddingsResponse)
async def generate_embeddings(input_data: SentencesInput):
    """Generate embeddings for a list of sentences without computing similarities."""
    if not tokenizer or not ort_session:
        raise HTTPException(status_code=500, detail="Model or tokenizer not initialized")
    
    embeddings = []
    for sentence in input_data.sentences:
        embedding = generate_embedding(sentence, tokenizer, ort_session)
        embeddings.append(embedding.tolist())
    
    return EmbeddingsResponse(
        embeddings=embeddings,
        sentences=input_data.sentences
    )

@app.post("/compute_similarity/", response_model=SimilarityResponse)
async def compute_similarity(input_data: SentencesInput):
    """Compute similarities between all pairs of input sentences."""
    if not tokenizer or not ort_session:
        raise HTTPException(status_code=500, detail="Model or tokenizer not initialized")
    
    # First generate embeddings
    embeddings = []
    for sentence in input_data.sentences:
        embedding = generate_embedding(sentence, tokenizer, ort_session)
        embeddings.append(embedding)
    
    # Calculate similarities between all pairs of sentences
    similarities = []
    for i in range(len(input_data.sentences)):
        for j in range(i + 1, len(input_data.sentences)):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append({
                "sentence1_index": i,
                "sentence2_index": j,
                "sentence1": input_data.sentences[i],
                "sentence2": input_data.sentences[j],
                "similarity_score": similarity
            })
    
    return SimilarityResponse(similarities=similarities)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 