from langchain_community.embeddings import HuggingFaceBgeEmbeddings

MODEL_NAME = "BAAI/bge-large-en"
DEVICE = 'cpu'

def get_embeddings():
    model_kwargs = {'device': DEVICE}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceBgeEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )