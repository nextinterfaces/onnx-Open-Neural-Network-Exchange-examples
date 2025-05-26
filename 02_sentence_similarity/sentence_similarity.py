from sentence_transformers import SentenceTransformer
import numpy as np

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    # Load the model directly
    print("Loading model from redis/langcache-embed-v1...")
    model = SentenceTransformer('redis/langcache-embed-v1')
    
    # Example sentences
    sentences = [
        'Will the value of Indian rupee increase after the ban of 500 and 1000 rupee notes?',
        'What will be the implications of banning 500 and 1000 rupees currency notes on Indian economy?',
        "Are Danish Sait's prank calls fake?",
    ]
    
    print("\nExample sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
    
    # Generate embeddings directly using the model
    print("\nGenerating embeddings...")
    embeddings = model.encode(sentences)
    
    for i, embedding in enumerate(embeddings):
        print(f"Generated embedding shape: {embedding.shape}")
    
    # Calculate similarities between sentences
    print("\nSimilarity scores between sentences:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            print(f"Similarity between sentence {i+1} and {j+1}: {similarity:.4f}")
    
    print("\nDone! The model successfully generated embeddings.")

if __name__ == "__main__":
    main() 