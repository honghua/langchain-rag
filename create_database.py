import os
import shutil

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pathlib import Path
from typing import List

from embeddings import get_embeddings

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def generate_data_store():
    documents = load_documents(Path(DATA_PATH))
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents(doc_dir: Path) -> List[Document]:
    documents = []
    for file_path in doc_dir.glob('*'):
        if file_path.is_file():
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": str(file_path)}
                )
                documents.append(doc)
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    hf = get_embeddings()

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, hf, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def main():
    generate_data_store()

if __name__ == "__main__":
    main()
