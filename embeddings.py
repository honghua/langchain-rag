import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


class Embeddings:
    MODEL_NAME = "BAAI/bge-large-en"
    DEVICE = 'cpu'

    @classmethod
    def get_embeddings(cls):
        model_kwargs = {'device': cls.DEVICE}
        encode_kwargs = {'normalize_embeddings': True}
        return HuggingFaceBgeEmbeddings(
            model_name=cls.MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

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

def evaluate_embeddings():
    # Get embedding for a word.
    embedding_function = Embeddings.get_embeddings()
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two sentence using custom cosine similarity evaluator
    evaluator = CosineSimilarityEvaluator(embedding_function)
    sentences = ("to be or not to be is a question", "The greatest glory in living lies not in never falling, but in rising every time we fall")
    similarity_score = evaluator.evaluate_string_pairs(prediction=sentences[0], prediction_b=sentences[1])
    print(f"Cosine similarity between '{sentences[0]}' and '{sentences[1]}': {similarity_score}")

if __name__ == "__main__":
    evaluate_embeddings()