import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="nexus_compliance_vault")

# Complex Dataset: Multiple versions and regions
documents = [
    # EU AI ACT - VERSION 1.0 (Outdated)
    "EU AI Act v1.0: AI systems for social scoring are permitted only for government use in national security.",
    # EU AI ACT - VERSION 2.0 (Current)
    "EU AI Act v2.0: AI systems for social scoring are STRICTLY PROHIBITED for all entities, including governments.",
    
    # GDPR PRIVACY
    "GDPR Article 22: Individuals have the right not to be subject to a decision based solely on automated processing.",
    "GDPR Data Residency: Data of EU citizens must be stored on servers located within the European Economic Area.",
    
    # US REGULATION (NIST)
    "NIST AI RMF: Organizations should prioritize 'Human-in-the-loop' systems for high-impact financial decisions.",
    "NIST Privacy Framework: De-identification is a key process for protecting individual privacy in large datasets.",
    
    # TECHNICAL IMPLEMENTATION (Best Practices)
    "Implementation Guide: When deploying RAG, use Recursive Character Splitting with a chunk size of 512 tokens.",
    "Security Protocol: All API keys must be rotated every 90 days and stored in a Hardware Security Module (HSM).",
    
    # CONFLICTING REGIONAL RULES
    "Regional Policy (India): AI deployment in healthcare requires a local data mirror within 30 days of launch.",
    "Regional Policy (UAE): AI systems in finance must be audited by a certified third-party every 6 months."
]

metadatas = [
    {"source": "EU_ACT", "version": "1.0", "status": "deprecated", "region": "EU"},
    {"source": "EU_ACT", "version": "2.0", "status": "active", "region": "EU"},
    {"source": "GDPR", "type": "legal", "topic": "privacy", "region": "EU"},
    {"source": "GDPR", "type": "technical", "topic": "storage", "region": "EU"},
    {"source": "NIST", "standard": "RMF", "topic": "governance", "region": "US"},
    {"source": "NIST", "standard": "Privacy", "topic": "data", "region": "US"},
    {"source": "RAG_GUIDE", "type": "technical", "topic": "engineering", "region": "global"},
    {"source": "SEC_DOC", "type": "security", "topic": "devops", "region": "global"},
    {"source": "IN_POLICY", "type": "local_law", "topic": "healthcare", "region": "IN"},
    {"source": "UAE_POLICY", "type": "local_law", "topic": "finance", "region": "UAE"}
]

ids = [f"nexus_{i:03d}" for i in range(len(documents))]

collection.add(documents=documents, metadatas=metadatas, ids=ids)
print("Complex Ingestion Complete.")