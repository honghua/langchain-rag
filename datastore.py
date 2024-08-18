import os
import shutil
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from embeddings import Embeddings

class DataStoreGenerator:
    CHROMA_PATH = "chroma"

    @classmethod
    def generate_data_store(cls, data_path: Path, overwrite: bool = False, chunk_size: int = 300, chunk_overlap: int = 100):
        if not overwrite and os.path.exists(cls.CHROMA_PATH):
            print(f"Data store already exists at {cls.CHROMA_PATH}. Use overwrite=True to recreate.")
            return

        documents = cls._load_documents(data_path)
        chunks = cls._split_text(documents, chunk_size, chunk_overlap)
        cls._save_to_chroma(chunks, overwrite)

    @classmethod
    def _load_documents(cls, doc_dir: Path) -> List[Document]:
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

    @classmethod
    def _split_text(cls, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        if chunks:
            document = chunks[10]
            print(document.page_content)
            print(document.metadata)
        return chunks

    @classmethod
    def _save_to_chroma(cls, chunks: List[Document], overwrite: bool):
        if overwrite and os.path.exists(cls.CHROMA_PATH):
            shutil.rmtree(cls.CHROMA_PATH)
        
        hf = Embeddings.get_embeddings()
        db = Chroma.from_documents(
            chunks, hf, persist_directory=cls.CHROMA_PATH
        )
        db.persist()
        print(f"Saved {len(chunks)} chunks to {cls.CHROMA_PATH}.")


if __name__ == "__main__":
    data_path = Path("data")
    DataStoreGenerator.generate_data_store(data_path, overwrite=True)