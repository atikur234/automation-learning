import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="nexus_compliance_vault")

print(f"--- VAULT DIAGNOSIS ---")
print(f"Total Chunks: {collection.count()}")

# Peek at the very first item to see its tags
if collection.count() > 0:
    sample = collection.peek(limit=1)
    print(f"Actual Metadata in DB: {sample['metadatas'][0]}")
else:
    print("❌ VAULT IS EMPTY. Re-run ingest_v2.py")