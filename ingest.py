import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

UPLOAD_DIR = "data/uploads"
VECTOR_DB_DIR = "data/faiss_index"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def ingest_documents(role: str, department: str):
    documents = []

    for file in os.listdir(UPLOAD_DIR):
        if file.endswith(".pdf"):
            file_path = os.path.join(UPLOAD_DIR, file)

            loader = PyPDFLoader(file_path)
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = file
                page.metadata["role"] = role
                page.metadata["department"] = department

            documents.extend(pages)

    if not documents:
        print("❌ No documents found to ingest.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    if os.path.exists(VECTOR_DB_DIR):
        db = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(VECTOR_DB_DIR)

    print("✅ Documents ingested successfully into FAISS.")
