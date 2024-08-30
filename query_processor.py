import os
import google.generativeai as genai
from dotenv import load_dotenv
from embeddings import Embeddings
from langchain_community.vectorstores import Chroma

# Define default values for generation parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.95

class QueryProcessor:
    CHROMA_PATH = "chroma"

    def __init__(self):
        self.embedding_function = Embeddings.get_embeddings()
        self.db = Chroma(persist_directory=self.CHROMA_PATH, embedding_function=self.embedding_function)

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

    def call_api(self, input_text):
        # Configure the API key
        load_dotenv()
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)

        # Set up generation configuration with default values
        generation_config = genai.types.GenerationConfig(
            temperature=DEFAULT_TEMPERATURE,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            stop_sequences=[],  # No stop sequences by default
            top_k=DEFAULT_TOP_K,
            top_p=DEFAULT_TOP_P
        )

        # Prepare text prompt
        text_prompt = [input_text] if input_text else []

        # Select the appropriate model (only text-based in this case)
        model_name = 'gemini-pro'
        model = genai.GenerativeModel(model_name)

        # Generate response from the model
        response = model.generate_content(
            text_prompt,
            stream=True,
            generation_config=generation_config
        )

        # Yield response chunks
        for message in response:
            yield message.text

    def process_query(self, query_text):
        results = self.search_db(query_text)
        if results is None:
            return "Unable to find matching results."

        context_text = self.format_context(results)
        messages = f"Answer the question based only on the following context: {context_text}\nQuestion: {query_text}"

        response_text = ''.join([chunk for chunk in self.call_api(messages)])

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = self.format_output(response_text, sources)
        print("context:\n", context_text)
        print(formatted_response)
        return response_text
