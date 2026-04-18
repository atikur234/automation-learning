import os
import time
from openai import OpenAI # Ollama uses the OpenAI-compatible local server
import chromadb

# 1. Setup Local Client
# No API Key needed, but 'ollama' is a placeholder
client_local = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama', 
)

def optimize_query(original_query):
    """Day 5: The Optimizer - Powered by Local Llama 3"""
    prompt = f"Rewrite this as a precise technical search query for a database. Output ONLY the query: {original_query}"
    
    response = client_local.chat.completions.create(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        temperature=0 # Keep it precise
    )
    return response.choices[0].message.content.strip()

def ask_nexus_research(query, region="EU"):
    # 2. Connect to Local DB
    client_db = chromadb.PersistentClient(path="./chroma_db")
    collection = client_db.get_collection(name="nexus_compliance_vault")

    # 3. Optimize (Unlimited calls!)
    search_term = optimize_query(query)
    print(f"DEBUG: [Local Optimizer: {search_term}]")

    # 4. Retrieve Chunks
    results = collection.query(
        query_texts=[search_term], 
        n_results=7, 
        where={"$and": [{"status": "active"}, {"region": region}]}
    )

    if not results['documents'][0]:
        return "No relevant data found in the local vault."

    # 5. Build Context
    context = ""
    for i, doc in enumerate(results['documents'][0]):
        source = results['metadatas'][0][i].get('source', 'Unknown')
        idx = results['metadatas'][0][i].get('chunk_index', 'N/A')
        context += f"--- SOURCE: {source} (Chunk {idx}) ---\n{doc}\n\n"

    # 6. The Judge - Final reasoning
    judge_prompt = f"""
    SYSTEM: You are a Senior Research Judge. Use the PROVIDED_CONTEXT to answer the USER_QUERY.
    RULES:
    - If the context doesn't have the answer, say 'Data missing in vault'.
    - Cite sources as [Source Name, Chunk #].
    
    PROVIDED_CONTEXT:
    {context}

    USER_QUERY: {query}
    """

    response = client_local.chat.completions.create(
        model="llama3",
        messages=[{"role": "user", "content": judge_prompt}]
    )
    return response.choices[0].message.content

# --- THE EXECUTION BLOCK ---
if __name__ == "__main__":
    try:
        print("\n🚀 STARTING LOCAL RESEARCH (OLLAMA)...")
        result = ask_nexus_research("Who are the co-chairs of the High-level Advisory Body?", region="EU")
        print("\n--- FINAL RESEARCH SUMMARY ---")
        print(result)
    except Exception as e:
        print(f"\n❌ SYSTEM ERROR: {e}")
        print("Check if Ollama is running in your taskbar!")