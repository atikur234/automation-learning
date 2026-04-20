import os
import time
import tiktoken
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# --- INITIALIZATION ---
client_local = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
tokenizer = tiktoken.get_encoding("cl100k_base")

print("🚀 Loading Reranker Model (Day 6 Engine)...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def count_tokens(text):
    return len(tokenizer.encode(text))

# --- PARALLEL SEARCH FUNCTIONS ---
def get_vector_search(query, collection, n):
    return collection.query(query_texts=[query], n_results=n)

def get_keyword_search(query, all_docs, n):
    bm25 = BM25Okapi([doc.lower().split() for doc in all_docs])
    return bm25.get_top_n(query.lower().split(), all_docs, n=n)

# --- THE MAIN RESEARCH ENGINE ---
def nexus_research_pro(query):
    # --- 1. SAFETY INITIALIZATION ---
    stats = {'retrieval_ms': 0, 'rerank_ms': 0, 'gen_ms': 0, 'total_ms': 0, 'tokens_in': 0, 'tokens_out': 0}
    start_total = time.time()
    
    client_db = chromadb.PersistentClient(path="./chroma_db")
    collection = client_db.get_collection(name="nexus_compliance_vault")
    all_data = collection.get()

    # --- 2. PARALLEL RETRIEVAL ---
    print(f"🔍 Searching vault for: '{query}'...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_v = executor.submit(get_vector_search, query, collection, 10)
        future_k = executor.submit(get_keyword_search, query, all_data['documents'], 10)
        vector_docs = future_v.result()['documents'][0]
        keyword_docs = future_k.result()
    stats['retrieval_ms'] = int((time.time() - t0) * 1000)

    # --- 3. RERANKING & SCORING ---
    t1 = time.time()
    candidates = list(set(vector_docs + keyword_docs))
    doc_to_meta = {doc: meta for doc, meta in zip(all_data['documents'], all_data['metadatas'])}
    enriched_pairs = [[query, f"SOURCE: {doc_to_meta.get(d, {}).get('source','')} | {d}"] for d in candidates]
    
    # We define scored_docs HERE so the if-statement can see it
    scored_docs = []
    if enriched_pairs:
        scores = reranker.predict(enriched_pairs)
        scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    stats['rerank_ms'] = int((time.time() - t1) * 1000)

    # --- 4. THE KILL SWITCH (Logic Check) ---
    if not scored_docs or scored_docs[0][1] < -4.0:
        stats['total_ms'] = int((time.time() - start_total) * 1000)
        return "❌ Data missing in vault (Relevance Threshold failed).", stats

    # --- 5. GENERATION ---
    t2 = time.time()
    best_context = "\n\n".join([doc for doc, score in scored_docs[:2]])
    prompt = f"SYSTEM: Use ONLY context.\nCONTEXT: {best_context}\n\nUSER: {query}"
    
    response = client_local.chat.completions.create(
        model="llama3", 
        messages=[{"role": "user", "content": prompt}],
        extra_body={"options": {"temperature": 0}}
    )
    
    answer = response.choices[0].message.content
    stats['gen_ms'] = int((time.time() - t2) * 1000)
    stats['tokens_in'] = count_tokens(prompt)
    stats['tokens_out'] = count_tokens(answer)
    stats['total_ms'] = int((time.time() - start_total) * 1000)
    
    return answer, stats

# --- THE LIVE DASHBOARD ---
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("==========================================")
    print("      NEXUS RESEARCH v4 PRO ONLINE        ")
    print("==========================================")
    
    while True:
        user_query = input("\n🔎 Enter Research Query (or 'exit'): ")
        if user_query.lower() == 'exit': break
        
        answer, stats = nexus_research_pro(user_query)
        
        print("\n" + "─" * 50)
        print(f"📝 ANSWER:\n{answer}")
        print("─" * 50)
        
        # Dashboard Table
        print("\n📊 TELEMETRY DASHBOARD")
        print(f"⏱️ Total Time:    {stats['total_ms']}ms")
        print(f"├─ Retrieval:    {stats.get('retrieval_ms', 0)}ms")
        print(f"├─ Reranking:    {stats.get('rerank_ms', 0)}ms")
        print(f"└─ Generation:   {stats.get('gen_ms', 0)}ms")
        print(f"🪙 Token Usage:   {stats.get('tokens_in', 0) + stats.get('tokens_out', 0)} (In: {stats.get('tokens_in', 0)} | Out: {stats.get('tokens_out', 0)})")
        print("==========================================")