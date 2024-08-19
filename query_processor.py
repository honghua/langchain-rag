import torch
import transformers
from langchain_community.vectorstores import Chroma
from embeddings import Embeddings

class QueryProcessor:
    CHROMA_PATH = "chroma"
    LLM_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    def __init__(self):
        self.embedding_function = Embeddings.get_embeddings()
        self.db = Chroma(persist_directory=self.CHROMA_PATH, embedding_function=self.embedding_function)
        self.pipeline = self._setup_pipeline()

    def _setup_pipeline(self):
        return transformers.pipeline(
            "text-generation",
            model=self.LLM_MODEL_ID,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def search_db(self, query_text, k=3, score_threshold=0.7):
        results = self.db.similarity_search_with_relevance_scores(query_text, k=k)
        if not results or results[0][1] < score_threshold:
            return None
        return results

    @staticmethod
    def format_context(results):
        return "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    @staticmethod
    def format_output(response_text, sources):
        return f"Response: {response_text}\nSources: {sources}"

    def process_query(self, query_text):
        results = self.search_db(query_text)
        if results is None:
            return "Unable to find matching results."

        context_text = self.format_context(results)
        messages = [
            {"role": "system", "content": f"Answer the question based only on the following context: {context_text}"}, 
            {"role": "user", "content": query_text},
        ]

        outputs = self.pipeline(messages, max_new_tokens=256)
        response_text = outputs[0]["generated_text"][-1]["content"]

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = self.format_output(response_text, sources)
        print("context:\n", context_text)
        print(formatted_response)
        return response_text

