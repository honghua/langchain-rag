import argparse
import torch
import transformers
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from embeddings import get_embeddings

CHROMA_PATH = "chroma"
MODEL_NAME = "BAAI/bge-large-en"
DEVICE = 'cpu'
LLM_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def search_db(db, query_text, k=3, score_threshold=0.7):
    results = db.similarity_search_with_relevance_scores(query_text, k=k)
    if not results or results[0][1] < score_threshold:
        return None
    return results

def format_context(results):
    return "\n\n---\n\n".join([doc.page_content for doc, _score in results])

def format_output(response_text, sources):
    return f"Response: {response_text}\nSources: {sources}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    embedding_function = get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = search_db(db, args.query_text)

    if results is None:
        print("Unable to find matching results.")
        return

    context_text = format_context(results)
    messages = [
        {"role": "system", "content": f"Answer the question based only on the following context: {context_text}"}, 
        {"role": "user", "content": args.query_text},
    ]
    print("context:\n", context_text)

    pipeline = transformers.pipeline(
        "text-generation",
        model=LLM_MODEL_ID,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    outputs = pipeline(messages, max_new_tokens=256)
    response_text = outputs[0]["generated_text"][-1]["content"]
    print(response_text)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = format_output(response_text, sources)
    print(formatted_response)

if __name__ == "__main__":
    main()