import chromadb
client = chromadb.PersistentClient(path="./chroma_db")

# Delete the old collection to start fresh
try:
    client.delete_collection(name="nexus_compliance_vault")
    print("🗑️ Vault cleared. Ready for fresh ingestion.")
except:
    print("Vault already empty.")