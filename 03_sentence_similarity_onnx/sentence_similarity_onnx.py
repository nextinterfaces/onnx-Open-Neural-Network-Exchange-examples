import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def mean_pooling(token_embeddings, attention_mask):
    """Perform mean pooling on token embeddings using attention mask."""
    # Convert attention mask to float and expand dimensions to match token embeddings
    input_mask_expanded = attention_mask[..., np.newaxis]
    # Sum embeddings weighted by attention mask
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    # Sum attention mask to get actual token count (excluding padding)
    sum_mask = np.clip(attention_mask.sum(axis=1), a_min=1e-9, a_max=None)
    # Calculate mean by dividing sum of embeddings by token count
    return sum_embeddings / sum_mask[:, np.newaxis]

def main():
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('redis/langcache-embed-v1')
    
    # Load ONNX model
    print("Loading ONNX model...")
    ort_session = ort.InferenceSession("../langcache-embed-v1-model.onnx")
    
    # Example sentences
    sentences = [
        'Will the value of Indian rupee increase after the ban of 500 and 1000 rupee notes?',
        'What will be the implications of banning 500 and 1000 rupees currency notes on Indian economy?',
        "Are Danish Sait's prank calls fake?",
    ]
    
    print("\nExample sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
    
    # Generate embeddings using ONNX model
    print("\nGenerating embeddings...")
    embeddings = []
    
    for sentence in sentences:
        # Tokenize the input
        inputs = tokenizer(
            sentence,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        
        # Run inference
        ort_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        
        # Get token embeddings
        token_embeddings = ort_session.run(None, ort_inputs)[0]
        
        # Apply mean pooling to get sentence embedding
        sentence_embedding = mean_pooling(token_embeddings, inputs['attention_mask'])
        embeddings.append(sentence_embedding[0])  # Get the first (and only) sentence embedding
        print(f"Generated embedding shape: {sentence_embedding[0].shape}")
    
    # Calculate similarities between sentences
    print("\nSimilarity scores between sentences:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            print(f"Similarity between sentence {i+1} and {j+1}: {similarity:.4f}")
    
    print("\nDone! The ONNX model successfully generated embeddings.")

if __name__ == "__main__":
    main() 