import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader

# 1. SETUP THE HIGH-PRECISION BRAIN
print("🧠 Initializing High-Precision Semantic Brain...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# We use 'standard_deviation' with a 1.25 threshold.
# This forces more 'cuts', creating smaller, more focused chunks.
text_splitter = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.25 
)

# 2. PDF PATH CHECKER
file_path = "data/un_ai_report.pdf"

if not os.path.exists(file_path):
    print(f"❌ ERROR: PDF not found at {os.path.abspath(file_path)}")
else:
    # 3. LOAD THE PDF
    print(f"📄 Loading PDF: {file_path}...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # 4. SEMANTIC SHREDDING
    print("✂️ Shredding PDF into High-Precision Atomic Chunks...")
    chunks = text_splitter.split_documents(pages)

    # 5. REBUILD THE VAULT
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try: 
        client.delete_collection("nexus_compliance_vault")
        print("🗑️ Old vault wiped.")
    except: 
        pass

    collection = client.create_collection(name="nexus_compliance_vault")

    print(f"📦 Indexing {len(chunks)} Atomic Semantic Chunks...")
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"sem_{i}"],
            documents=[chunk.page_content],
            metadatas=[{
                "source": "un_ai_report.pdf", 
                "type": "semantic",
                "page": chunk.metadata.get("page", 0) + 1 
            }]
        )

    print(f"✅ Success! Created {len(chunks)} high-precision chunks.")