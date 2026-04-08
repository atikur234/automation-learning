import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# 1. Setup the Smart Splitter
# chunk_size: characters per piece. chunk_overlap: context bridge.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]
)

def process_document(file_path, region="global"):
    # A. Extract Text
    text = ""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text()
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    # B. Transformation (Chunking)
    chunks = text_splitter.split_text(text)
    print(f"📄 Processed {file_path}: Created {len(chunks)} chunks.")

    # C. Loading (To ChromaDB)
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="nexus_compliance_vault")

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        # Metadata is the 'tag' for filtering we used in Day 3
        metadatas.append({
            "source": os.path.basename(file_path),
            "region": region,
            "status": "active",
            "chunk_index": i
        })
        ids.append(f"{os.path.basename(file_path)}_{i}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ Successfully ingested {file_path} into the vault.")

# --- RUN THE PIPELINE ---
if __name__ == "__main__":
    folder_path = "./data_sources"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # For now, let's tag everything in this folder as 'EU' for testing
        process_document(file_path, region="EU")