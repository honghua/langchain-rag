import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from embeddings import get_embeddings

class CosineSimilarityEvaluator:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
    
    def evaluate_string_pairs(self, prediction: str, prediction_b: str) -> float:
        # Get the embeddings for the two strings
        vector_a = self.embedding_function.embed_query(prediction)
        vector_b = self.embedding_function.embed_query(prediction_b)
        
        # Reshape to (1, -1) because cosine_similarity expects 2D arrays
        vector_a = np.array(vector_a).reshape(1, -1)
        vector_b = np.array(vector_b).reshape(1, -1)
        
        # Compute the cosine similarity
        similarity = cosine_similarity(vector_a, vector_b)[0][0]
        
        return similarity

def main():
    # Get embedding for a word.
    embedding_function = get_embeddings()
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words using custom cosine similarity evaluator
    evaluator = CosineSimilarityEvaluator(embedding_function)
    # words = ("apple", "iphone")
    words = ("to be or not to be is a question", "car is running")
    similarity_score = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Cosine similarity between '{words[0]}' and '{words[1]}': {similarity_score}")

if __name__ == "__main__":
    main()
