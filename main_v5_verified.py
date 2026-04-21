import os
import time
import tiktoken
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# --- 1. INITIALIZATION ---
client_local = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
tokenizer = tiktoken.get_encoding("cl100k_base")

print("🚀 Loading High-Precision Reranker...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def count_tokens(text):
    return len(tokenizer.encode(text))

# --- 2. SEARCH ENGINE MODULES ---
def get_vector_search(query, collection, n):
    return collection.query(query_texts=[query], n_results=n)

def get_keyword_search(query, all_docs, n):
    bm25 = BM25Okapi([doc.lower().split() for doc in all_docs])
    return bm25.get_top_n(query.lower().split(), all_docs, n=n)

# --- 3. THE VERIFIED RESEARCH ENGINE ---
def nexus_research_v5(query):
    stats = {'retrieval_ms': 0, 'rerank_ms': 0, 'gen_ms': 0, 'total_ms': 0, 'tokens_in': 0, 'tokens_out': 0}
    start_total = time.time()
    
    # Connect to Vault
    client_db = chromadb.PersistentClient(path="./chroma_db")
    collection = client_db.get_collection(name="nexus_compliance_vault")
    all_data = collection.get()

    # A. PARALLEL RETRIEVAL (Narrow Funnel - Top 5)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_v = executor.submit(get_vector_search, query, collection, 5)
        f_k = executor.submit(get_keyword_search, query, all_data['documents'], 5)
        vector_docs = f_v.result()['documents'][0]
        keyword_docs = f_k.result()
    stats['retrieval_ms'] = int((time.time() - t0) * 1000)

    # B. RERANKING & EVIDENCE MAPPING
    t1 = time.time()
    candidates = list(set(vector_docs + keyword_docs))
    
    # Map documents to their metadata for citation
    doc_to_meta = {doc: meta for doc, meta in zip(all_data['documents'], all_data['metadatas'])}
    
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    scored_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    stats['rerank_ms'] = int((time.time() - t1) * 1000)

    # C. THRESHOLD GATE (The Hallucination Guard)
    if not scored_results or scored_results[0][1] < -4.0:
        stats['total_ms'] = int((time.time() - start_total) * 1000)
        return "❌ Data missing in vault.", None, stats

    # D. CONTEXT PREPARATION (Atomic Top-1)
    best_doc = scored_results[0][0]
    meta = doc_to_meta.get(best_doc, {})
    
    evidence = {
        "file": meta.get("source", "Unknown"),
        "page": meta.get("page", "N/A"),
        "snippet": best_doc[:300].replace("\n", " ") + "..." 
    }

    # E. SNIPER GENERATION
    t2 = time.time()
    # OLD: prompt = f"### Instruction: Answer precisely using the context below. Include a short direct quote for proof..."

# NEW (The Professional Auditor):
    prompt = (
        f"### Instruction: Answer the question using the context. \n"
        f"1. Be extremely brief (bullet points).\n"
        f"2. Provide a 'Proof Quote' no longer than 10 words.\n\n"
        f"### Context:\n{best_doc}\n\n"
        f"### Question:\n{query}\n\n"
        "### Answer:"
     )
    
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
    
    return answer, evidence, stats

# --- 4. TERMINAL DASHBOARD ---
if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("══ NexusResearch v5: Verified Intelligence ══")
    
    while True:
        user_query = input("\n🔎 Research Query: ")
        if user_query.lower() in ['exit', 'quit']: break
        
        answer, evidence, stats = nexus_research_v5(user_query)
        
        print("\n" + "━"*60)
        print(f"📝 ANSWER:\n{answer}")
        
        if evidence:
            print("\n🛡️ VERIFICATION:")
            print(f"  📍 Source: {evidence['file']} | Page: {evidence['page']}")
            print(f"  📜 Snippet: {evidence['snippet']}")
        
        print("━"*60)
        print(f"📊 {stats['total_ms']}ms | In:{stats['tokens_in']} Out:{stats['tokens_out']} | Rerank:{stats['rerank_ms']}ms")
        print("━"*60)

        # --- NEW: PERSISTENT AUDIT LOG ---
        with open("nexus_audit_log.md", "a", encoding="utf-8") as f:
            f.write(f"## Query: {user_query}\n")
            f.write(f"**Answer:** {answer}\n")
            f.write(f"**Source:** {evidence['file']} (Page {evidence['page']})\n")
            f.write(f"**Verification Snippet:** `{evidence['snippet']}`\n")
            f.write(f"**Performance:** {stats['total_ms']}ms\n\n")
            f.write("---\n")
        print("💾 Research saved to nexus_audit_log.md")