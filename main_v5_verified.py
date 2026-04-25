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

print("🚀 Initializing Nexus Synthesis Engine (Day 10 Final)...")
# Re-enabling GPU for the RTX 5050 Blackwell Architecture
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
def count_tokens(text):
    return len(tokenizer.encode(text))

# --- 2. SEARCH ENGINE MODULES ---
def get_vector_search(query, collection, n):
    return collection.query(query_texts=[query], n_results=n)

def get_keyword_search(query, all_docs, n):
    bm25 = BM25Okapi([doc.lower().split() for doc in all_docs])
    return bm25.get_top_n(query.lower().split(), all_docs, n=n)

# --- 3. THE EXECUTIVE RESEARCH ENGINE ---
def nexus_research_final(query):
    # Added tokens_out to stats
    stats = {'retrieval_ms': 0, 'rerank_ms': 0, 'gen_ms': 0, 'total_ms': 0, 'tokens_in': 0, 'tokens_out': 0, 'rerank_score': 0}
    start_total = time.time()
    
    client_db = chromadb.PersistentClient(path="./chroma_db")
    collection = client_db.get_collection(name="nexus_compliance_vault")
    all_data = collection.get()

    # A. PARALLEL RETRIEVAL
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_v = executor.submit(get_vector_search, query, collection, 5)
        f_k = executor.submit(get_keyword_search, query, all_data['documents'], 5)
        v_docs = f_v.result()['documents'][0]
        k_docs = f_k.result()
    stats['retrieval_ms'] = int((time.time() - t0) * 1000)

    # B. RERANKING & EVIDENCE MAPPING
    t1 = time.time()
    candidates = list(set(v_docs + k_docs))
    doc_to_meta = {doc: meta for doc, meta in zip(all_data['documents'], all_data['metadatas'])}
    
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    scored_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    stats['rerank_ms'] = int((time.time() - t1) * 1000)

    if not scored_results or scored_results[0][1] < -4.0:
        return "❌ Data missing in vault.", None, stats

    # --- DAY 10 SYNTHESIS LOGIC: TOP-2 FOR COMPARISONS ---
    best_score = scored_results[0][1]
    stats['rerank_score'] = best_score
    
    comparison_keywords = ["compare", "difference", "versus", "both", "opportunity scan"]
    if any(k in query.lower() for k in comparison_keywords) and len(scored_results) > 1:
        best_doc = f"--- CONTEXT A ---\n{scored_results[0][0]}\n\n--- CONTEXT B ---\n{scored_results[1][0]}"
        primary_meta = doc_to_meta.get(scored_results[0][0], {})
    else:
        best_doc = scored_results[0][0]
        primary_meta = doc_to_meta.get(best_doc, {})
    
    evidence = {
        "file": primary_meta.get("source", "Unknown"),
        "page": primary_meta.get("page", "N/A"),
        "snippet": scored_results[0][0][:250].replace("\n", " ") + "..." 
    }

    # D. EXECUTIVE SYSTEM PROMPT
    t2 = time.time()
    prompt = (
        f"### IDENTITY: You are Nexus Research Intelligence. Provide a precise executive report.\n"
        f"### INSTRUCTIONS: \n"
        f"1. Summarize key facts in bullet points.\n"
        f"2. Compare specific sections if multiple contexts are provided.\n"
        f"3. Maintain table structures and provide a proof quote.\n\n"
        f"### CONTEXT:\n{best_doc}\n\n"
        f"### QUESTION:\n{query}\n\n"
        f"### EXECUTIVE REPORT:"
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

# --- 4. THE GRAND FINALE UI ---
if __name__ == "__main__":
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*64)
        print("   N E X U S   R E S E A R C H   I N T E L L I G E N C E")
        print("="*64)
        
        user_query = input("\n🔎 ENTER RESEARCH QUERY (or 'exit'): ")
        if user_query.lower() in ['exit', 'quit']: break
        
        print("\n📡 Processing Signal...")
        answer, evidence, stats = nexus_research_final(user_query)
        
        conf = min(100, max(0, int((stats['rerank_score'] + 4) * 20))) 

        print("\n" + "━"*64)
        print(f"📝 EXECUTIVE REPORT:\n{answer}")
        print("━"*64)
        
        if evidence:
            print(f"✅ TRUST VERIFICATION")
            print(f"   📍 SOURCE: {evidence['file']} | PAGE: {evidence['page']}")
            print(f"   🎯 CONFIDENCE: {conf}%")
            print(f"   📜 PROOF: \"{evidence['snippet']}\"")
            
            with open("nexus_audit_log.md", "a", encoding="utf-8") as f:
                f.write(f"### Query: {user_query}\n**Report:** {answer}\n**Page:** {evidence['page']}\n**Conf:** {conf}%\n---\n")
        
        # Telemetry now includes Out tokens for efficiency tracking
        print(f"\n📊 TELEMETRY: {stats['total_ms']}ms")
        print(f"├─ Tokens: In:{stats['tokens_in']} | Out:{stats['tokens_out']}")
        print(f"└─ Latency: Rerank:{stats['rerank_ms']}ms | Gen:{stats['gen_ms']}ms")
        print("="*64)
        input("\nPress Enter for next query...")