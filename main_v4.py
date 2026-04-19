import os
import time
import tiktoken
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# 1. SETUP & INITIALIZATION
client_local = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
# Use cl100k_base (compatible with Llama 3 / GPT-4 tokenization styles)
tokenizer = tiktoken.get_encoding("cl100k_base")

# Load Reranker once at startup
print("🚀 Loading Reranker Model...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def count_tokens(text):
    return len(tokenizer.encode(text))

def get_hybrid_context_v4(query, collection, n_initial=15):
    """Hybrid Search with Metadata Injection and Latency Tracking"""
    timers = {}
    
    # A. VECTOR SEARCH
    t0 = time.time()
    vector_results = collection.query(query_texts=[query], n_results=n_initial)
    vector_docs = vector_results['documents'][0]
    timers['vector_search'] = round(time.time() - t0, 3)
    
    # B. KEYWORD SEARCH (BM25)
    t1 = time.time()
    all_data = collection.get()
    all_docs = all_data['documents']
    all_metas = all_data['metadatas']
    
    bm25 = BM25Okapi([doc.lower().split() for doc in all_docs])
    keyword_docs = bm25.get_top_n(query.lower().split(), all_docs, n=n_initial)
    timers['keyword_search'] = round(time.time() - t1, 3)
    
    # C. RERANKING
    t2 = time.time()
    candidate_docs = list(set(vector_docs + keyword_docs))
    doc_to_meta = {doc: meta for doc, meta in zip(all_docs, all_metas)}
    
    enriched_pairs = []
    for doc in candidate_docs:
        meta = doc_to_meta.get(doc, {})
        source = meta.get('source', 'Unknown')
        enriched_text = f"SOURCE: {source} | CONTENT: {doc}"
        enriched_pairs.append([query, enriched_text])
    
    scores = reranker.predict(enriched_pairs)
    scored_docs = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
    timers['reranking'] = round(time.time() - t2, 3)
    
    return [doc for doc, score in scored_docs[:3]], timers

def ask_nexus_v4(query):
    report = {"query": query, "steps": {}}
    start_all = time.time()
    
    # Connect to DB
    client_db = chromadb.PersistentClient(path="./chroma_db")
    collection = client_db.get_collection(name="nexus_compliance_vault")

    # 1. RETRIEVAL & RERANKING
    best_chunks, retrieval_timers = get_hybrid_context_v4(query, collection)
    report['steps'].update(retrieval_timers)
    
    context = "\n\n".join(best_chunks)
    
    # 2. GENERATION
    t3 = time.time()
    judge_prompt = f"SYSTEM: Use ONLY the context to answer.\nCONTEXT:\n{context}\n\nUSER: {query}"
    
    response = client_local.chat.completions.create(
        model="llama3",
        messages=[{"role": "user", "content": judge_prompt}]
    )
    answer = response.choices[0].message.content
    report['steps']['generation'] = round(time.time() - t3, 3)
    
    # 3. TOKEN STATS
    report['tokens'] = {
        "input": count_tokens(judge_prompt),
        "output": count_tokens(answer),
        "total": count_tokens(judge_prompt) + count_tokens(answer)
    }
    
    report['total_latency'] = round(time.time() - start_all, 3)
    
    return answer, report

if __name__ == "__main__":
    print("\n🕵️ STARTING NEXUS OBSERVER (v4)...")
    q = "Who are the co-chairs of the High-level Advisory Body?"
    answer, stats = ask_nexus_v4(q)
    
    print("-" * 30)
    print(f"ANSWER: {answer}")
    print("-" * 30)
    print("\n📊 SYSTEM TELEMETRY:")
    print(f"Total Latency: {stats['total_latency']}s")
    print(f"├─ Retrieval (Vector): {stats['steps']['vector_search']}s")
    print(f"├─ Retrieval (BM25):   {stats['steps']['keyword_search']}s")
    print(f"├─ Reranking:          {stats['steps']['reranking']}s")
    print(f"└─ Generation:         {stats['steps']['generation']}s")
    print(f"\n🪙 TOKEN USAGE: {stats['tokens']['total']} (In: {stats['tokens']['input']} | Out: {stats['tokens']['output']})")