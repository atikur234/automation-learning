import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader

# 1. SETUP THE BRAIN (Now using the updated library)
print("🧠 Initializing Modern Semantic Brain...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Percentile split: It cuts where topic similarity drops significantly
text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

# 2. PDF PATH CHECKER
# Ensure your file is in 'D:\NexusResearch\data ingestion\data\un_ai_report.pdf'
file_path = "data/un_ai_report.pdf"

if not os.path.exists(file_path):
    print(f"❌ ERROR: PDF not found at {os.path.abspath(file_path)}")
    print("👉 ACTION: Make sure the file is named 'un_ai_report.pdf' inside the 'data' folder.")
else:
    # 3. LOAD THE PDF
    print(f"📄 Loading PDF: {file_path}...")
    loader = PyPDFLoader(file_path)
    # PyPDFLoader splits by page initially
    pages = loader.load()

    # 4. SEMANTIC SHREDDING
    print("✂️ Shredding PDF based on Topic Shifts (Processing meaning)...")
    # This takes the text from all pages and regroup them by 'meaning'
    chunks = text_splitter.split_documents(pages)

    # 5. REBUILD THE VAULT
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Wipe old data so we don't have 'duplicate' or 'messy' indices
    try: 
        client.delete_collection("nexus_compliance_vault")
        print("🗑️ Old vault wiped.")
    except: 
        pass

    collection = client.create_collection(name="nexus_compliance_vault")

    print(f"📦 Indexing {len(chunks)} high-quality semantic chunks...")
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"sem_{i}"],
            documents=[chunk.page_content],
            metadatas=[{
                "source": "un_ai_report.pdf", 
                "type": "semantic",
                "page": chunk.metadata.get("page", 0) + 1 # Tracks actual PDF page!
            }]
        )

    print(f"✅ Success! Created {len(chunks)} semantic chunks.")