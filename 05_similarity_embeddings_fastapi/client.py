import requests
import json
from typing import List, Dict
import numpy as np

class SentenceEmbeddingsClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def generate_embeddings(self, sentences: List[str]) -> Dict:
        """
        Generate embeddings for a list of sentences.
        
        Args:
            sentences: List of sentences to generate embeddings for
            
        Returns:
            Dictionary containing embeddings and original sentences
        """
        response = requests.post(
            f"{self.base_url}/generate_embeddings/",
            json={"sentences": sentences}
        )
        response.raise_for_status()
        return response.json()
    
    def compute_similarity(self, sentences: List[str]) -> Dict:
        """
        Compute similarities between all pairs of sentences.
        
        Args:
            sentences: List of sentences to compare
            
        Returns:
            Dictionary containing similarity scores for all pairs
        """
        response = requests.post(
            f"{self.base_url}/compute_similarity/",
            json={"sentences": sentences}
        )
        response.raise_for_status()
        return response.json()

def main():
    # Example sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn fox leaps above the sleepy hound",
        "The weather is beautiful today",
        "It's a lovely sunny day outside"
    ]
    
    client = SentenceEmbeddingsClient()
    
    # Example 1: Generate embeddings only
    print("\n=== Generating Embeddings ===")
    embeddings_result = client.generate_embeddings(sentences)
    print(f"Generated embeddings for {len(embeddings_result['embeddings'])} sentences")
    print(f"Embedding dimension: {len(embeddings_result['embeddings'][0])}")
    
    # Example 2: Compute similarities
    print("\n=== Computing Similarities ===")
    similarities_result = client.compute_similarity(sentences)
    
    # Print similarity scores in a readable format
    print("\nSimilarity Scores:")
    for pair in similarities_result['similarities']:
        print(f"\nPair {pair['sentence1_index']} - {pair['sentence2_index']}:")
        print(f"Sentence 1: {pair['sentence1']}")
        print(f"Sentence 2: {pair['sentence2']}")
        print(f"Similarity Score: {pair['similarity_score']:.4f}")

if __name__ == "__main__":
    main() 