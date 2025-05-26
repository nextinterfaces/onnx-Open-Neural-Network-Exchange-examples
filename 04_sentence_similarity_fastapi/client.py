import requests
import json
from typing import List
from pprint import pprint

def test_sentence_similarity(url: str, sentences: List[str]) -> None:
    """
    Test the sentence similarity API with the given sentences.
    
    Args:
        url: The API endpoint URL
        sentences: List of sentences to compare
    """
    print("\n" + "="*80)
    print("Testing with sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
    
    try:
        response = requests.post(
            url,
            json={"sentences": sentences},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        
        result = response.json()
        
        print("\nEmbeddings shape for each sentence:")
        for i, embedding in enumerate(result["embeddings"], 1):
            print(f"Sentence {i}: {len(embedding)} dimensions")
        
        print("\nSimilarity scores:")
        for pair in result["similarities"]:
            print(f"\nPair {pair['sentence1_index']+1} and {pair['sentence2_index']+1}:")
            print(f"Similarity score: {pair['similarity_score']:.4f}")
            print(f"Sentence 1: {pair['sentence1']}")
            print(f"Sentence 2: {pair['sentence2']}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except json.JSONDecodeError:
        print("Error decoding JSON response")
    except KeyError as e:
        print(f"Missing key in response: {e}")
    
    print("="*80 + "\n")

def main():
    base_url = "http://localhost:8000"
    endpoint = f"{base_url}/compute_similarity/"
    
    # Test case 1: Similar sentences about currency
    test_sentence_similarity(endpoint, [
        "Will the value of Indian rupee increase after the ban of 500 and 1000 rupee notes?",
        "What will be the implications of banning 500 and 1000 rupees currency notes on Indian economy?",
        "Are Danish Sait's prank calls fake?"
    ])
    
    # Test case 2: Similar sentences about technology
    test_sentence_similarity(endpoint, [
        "What are the best practices for machine learning model deployment?",
        "How do you deploy ML models in production environments?",
        "What is the capital of France?"
    ])
    
    # Test case 3: Similar sentences about food
    test_sentence_similarity(endpoint, [
        "What's the best way to cook pasta al dente?",
        "How do you make perfect Italian pasta?",
        "What's the secret to cooking perfect pasta every time?"
    ])
    
    # Test case 4: Edge case - Single word sentences
    test_sentence_similarity(endpoint, [
        "Hello",
        "Hi",
        "Greetings"
    ])
    
    # Test case 5: Edge case - Long sentences
    test_sentence_similarity(endpoint, [
        "The quick brown fox jumps over the lazy dog while the sun sets in the horizon, painting the sky with beautiful shades of orange and purple, creating a magnificent spectacle for all to see.",
        "As the sun descends below the horizon, it casts a warm glow across the landscape, where a swift brown fox can be seen leaping over a drowsy dog, creating a picturesque scene.",
        "In the twilight hours, a fox of brown coloration rapidly propels itself over a canine displaying characteristics of lethargy, all while the celestial body known as the sun makes its descent."
    ])

if __name__ == "__main__":
    main() 