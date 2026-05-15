import os
import chromadb
import pdfplumber
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

# 1. SETUP THE BRAIN
print("🧠 Initializing Multi-Doc Structural Brain...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.25 
)

# 2. SOURCE CONFIGURATION
# Point this to your folder containing BOTH the UN and EU PDFs
input_folder = "data/" 
client = chromadb.PersistentClient(path="./chroma_db")

# --- DAY 13 MODIFICATION: PERSISTENT COLLECTION ---
# We use get_or_create so we don't wipe existing documents
collection = client.get_or_create_collection(name="nexus_compliance_vault")

# 3. MULTI-FILE LOOP
for filename in os.listdir(input_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(input_folder, filename)
        all_extracted_docs = []
        
        print(f"\n📄 Analyzing Structure of {filename}...")
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract Tables
                tables = page.extract_tables()
                table_content = ""
                if tables:
                    for table in tables:
                        rows = [" | ".join([str(cell).replace('\n', ' ') if cell else "" for cell in row]) for row in table]
                        table_content += "\n| " + " |\n| ".join(rows) + " |\n"

                # Extract Text
                text_content = page.extract_text() or ""
                full_page_content = f"--- SOURCE: {filename} | PAGE {i+1} ---\n{table_content}\n{text_content}"
                
                # Wrap for splitter
                all_extracted_docs.append(Document(page_content=full_page_content, metadata={"page": i+1, "source": filename}))

        # 4. SEMANTIC SHREDDING
        print(f"✂️ Shredding {filename} into chunks...")
        chunks = text_splitter.split_documents(all_extracted_docs)

        # 5. THE CONTEXTUAL HANDSHAKE
        print(f"🧵 Stitching boundaries for {filename}...")
        for i in range(len(chunks)):
            if i > 0:
                prefix = chunks[i-1].page_content[-300:] + " [...] "
                chunks[i].page_content = prefix + chunks[i].page_content

        # 6. INCREMENTAL INDEXING
        print(f"📦 Indexing {len(chunks)} chunks into the vault...")
        # Use timestamp to ensure unique IDs across multiple documents
        timestamp = int(time.time())
        for i, chunk in enumerate(chunks):
            collection.add(
                ids=[f"{filename}_{timestamp}_{i}"],
                documents=[chunk.page_content],
                metadatas=[{
                    "source": filename, 
                    "type": "structural_semantic",
                    "page": chunk.metadata.get("page", 0)
                }]
            )

print(f"\n✅ DAY 13 INGESTION COMPLETE.")
print(f"Total documents now in vault: {collection.count()}")