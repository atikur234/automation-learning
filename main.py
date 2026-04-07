import os
import time
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import errors

# 1. Configuration
load_dotenv()
client_ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_answer_from_vault(user_query, region="EU"):
    """
    Retrieves grounded context from ChromaDB and generates a response.
    """
    # 2. Database Connection
    client_db = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        collection = client_db.get_collection(name="nexus_compliance_vault")
    except Exception:
        return "Error: Collection not found. Please run ingest.py first."

    # 3. Filtered Retrieval (Active versions + Region-specific)
    # Using the logic: (Status is Active) AND (Region is User-Region OR Global)
    results = collection.query(
        query_texts=[user_query],
        n_results=2,
        where={
            "$and": [
                {"status": "active"}, 
                {"$or": [{"region": region}, {"region": "global"}]}
            ]
        }
    )

    if not results['documents'][0]:
        return f"I lack sufficient data in my internal database for the {region} region."

    # 4. Context Assembly with Source Citations
    context = ""
    for i, doc in enumerate(results['documents'][0]):
        source = results['metadatas'][0][i].get('source', 'Unknown')
        context += f"SOURCE [{source}]: {doc}\n\n"

    # 5. Strict Grounding Prompt
    prompt = f"""
    SYSTEM: You are a Senior Research AI. 
    Answer the USER_QUERY using ONLY the PROVIDED_CONTEXT. 
    If the context is insufficient, state that you cannot answer.
    Always cite the [SOURCE] for every fact mentioned.

    PROVIDED_CONTEXT:
    {context}

    USER_QUERY: {user_query}
    """

    # 6. Generation with Automatic Retry (Handling Rate Limits)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client_ai.models.generate_content(
                model="gemini-2.5-flash-lite", 
                contents=prompt
            )
            return response.text
        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60 * (attempt + 1)
                print(f"⚠️ Rate limit hit. Waiting {wait_time}s to retry...")
                time.sleep(wait_time)
            else:
                return f"API Error: {e}"
    
    return "Error: Maximum retries reached due to rate limits."

# --- TEST EXECUTION ---
if __name__ == "__main__":
    print("\n--- Initializing Nexus-Research Reasoning Engine ---")
    
    # Test Case 1: EU Active Law
    print("\n[Querying EU Scope...]")
    print(get_answer_from_vault("What are the rules for social scoring?", region="EU"))

    # Test Case 2: Out-of-Scope Query
    print("\n[Querying US Scope...]")
    print(get_answer_from_vault("What are the rules for social scoring?", region="US"))