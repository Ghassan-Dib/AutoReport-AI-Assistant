import os
import fitz
import logging
from langchain_core.documents import Document
from docx import Document as DocxReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    EMBEDDINGS_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_STORE_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text("text")
            text = text.replace("\n", " ").replace("  ", " ").strip()
    return text


def extract_text_from_docx(path):
    doc = DocxReader(path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return text


def load_documents_from_directory(root_dir):
    documents = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif file.endswith(".docx"):
                text = extract_text_from_docx(file_path)
            else:
                continue

            if text.strip():
                documents.append(
                    Document(page_content=text, metadata={"source": file_path})
                )
    return documents


def build_vectorstore(data_root="data/", db_path=VECTOR_STORE_PATH):
    logging.info("\nLoading documents from folders...")
    docs = load_documents_from_directory(data_root)
    logging.info(f"✅ Loaded {len(docs)} documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    logging.info(f"✅ Split into {len(chunks)} chunks.")

    logging.info("\nCreating embeddings using Hugging Face model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    logging.info("\nBuilding FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    vectorstore.save_local(db_path)
    logging.info(f"✅ Vectorstore saved to {db_path}")


if __name__ == "__main__":
    build_vectorstore(data_root="data")
