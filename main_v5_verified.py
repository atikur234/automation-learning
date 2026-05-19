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

print("🚀 Initializing Nexus 'Judge' Engine (Day 14)...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')

def count_tokens(text):
    return len(tokenizer.encode(text))

# --- 2. SEARCH ENGINE MODULES ---
def get_vector_search(query, collection, n):
    return collection.query(query_texts=[query], n_results=n)

def get_keyword_search(query, all_docs, n):
    bm25 = BM25Okapi([doc.lower().split() for doc in all_docs])
    return bm25.get_top_n(query.lower().split(), all_docs, n=n)

# --- 3. AUDITORS & JUDGES ---
def agentic_audit(query, current_answer):
    audit_prompt = (
       f"### IDENTITY: Senior Inter-Nexus Auditor.\n"
       f"### TASK: If a query involves a ban or prohibition, the Judge must find the specific Article (EU) or Recommendation Number (UN). If no specific number is found, label the data as 'High Uncertainty / General Philosophy only'. "
       f"If Article numbers or fine percentages are missing, you MUST pivot.\n"
       f"### QUERY: {query}\n"
       f"### ANSWER: {current_answer}\n\n"
       f"INSTRUCTION: Output 'NEED: [topic]' or 'COMPLETE'."
     )
    response = client_local.chat.completions.create(
        model="llama3", 
        messages=[{"role": "user", "content": audit_prompt}],
        extra_body={"options": {"temperature": 0}}
    )
    return response.choices[0].message.content.strip()

# NEW: DAY 14 CONFLICT DETECTOR
def detect_conflicts(context_text, query):
    """Analyzes retrieved context for direct contradictions between sources."""
    conflict_prompt = (
        f"### IDENTITY: Judicial Research Auditor.\n"
        f"### TASK: Analyze context for CONTRADICTIONS between sources (e.g. UN vs EU).\n"
        f"### CONTEXT:\n{context_text}\n"
        f"### QUERY: {query}\n\n"
        f"INSTRUCTION: If sources disagree on a fact or mandate, list the conflict clearly. "
        f"If they agree, output 'CONSISTENT'."
    )
    res = client_local.chat.completions.create(
        model="llama3", 
        messages=[{"role": "user", "content": conflict_prompt}],
        extra_body={"options": {"temperature": 0}}
    )
    return res.choices[0].message.content.strip()

# --- 3.5 RECURSIVE SUMMARY (UPDATED) ---
def get_recursive_summary(collection, page_number, source_name):
    page_range = [max(1, page_number - 1), page_number, page_number + 1]
    results = collection.get(where={"$and": [{"page": {"$in": page_range}}, {"source": source_name}]})
    
    if not results['documents']: return "No broader chapter context found."
    
    full_text = " ".join(results['documents'])
    summary_prompt = (
        f"### TASK: Analyze the NARRATIVE ARC of these pages in {source_name}.\n"
        f"### TEXT:\n{full_text[:6000]}\n\n"
        f"### SUMMARY (Focus on 'Problem -> Solution' logic):"
    )
    res = client_local.chat.completions.create(
        model="llama3", 
        messages=[{"role": "user", "content": summary_prompt}],
        extra_body={"options": {"temperature": 0}}
    )
    return res.choices[0].message.content

# --- 4. THE EXECUTIVE RESEARCH ENGINE ---
def nexus_research_final(query):
    stats = {'total_ms': 0, 'tokens_in': 0, 'tokens_out': 0, 'rerank_score': 0, 'agentic_pivots': 0, 'recursive_summaries': 0, 'files_consulted': []}
    start_total = time.time()
    
    client_db = chromadb.PersistentClient(path="./chroma_db")
    collection = client_db.get_collection(name="nexus_compliance_vault")
    all_data = collection.get()

    # A. RETRIEVAL & RERANK
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_v = executor.submit(get_vector_search, query, collection, 8) 
        f_k = executor.submit(get_keyword_search, query, all_data['documents'], 8)
        v_docs = f_v.result()['documents'][0]
        k_docs = f_k.result()
    
    candidates = list(set(v_docs + k_docs))
    doc_to_meta = {doc: meta for doc, meta in zip(all_data['documents'], all_data['metadatas'])}
    
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    scored_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    if not scored_results or scored_results[0][1] < -5.0:
        return "❌ Data missing in vault.", None, stats

    stats['rerank_score'] = scored_results[0][1]

    # B. SOURCE ROUTING & CONFLICT DETECTION
    context_blocks = []
    seen_sources = set()
    for doc, score in scored_results[:6]:
        meta = doc_to_meta.get(doc, {})
        source = meta.get("source", "Unknown")
        seen_sources.add(source)
        context_blocks.append(f"--- SOURCE: {source} (Page {meta.get('page', 'N/A')}) ---\n{doc}")

    stats['files_consulted'] = list(seen_sources)
    full_context = "\n\n".join(context_blocks)
    
    # NEW: Run Conflict Judge
    print("⚖️  JUDGE PASS: Checking for source contradictions...")
    discrepancy_report = detect_conflicts(full_context, query)

    # C. GENERATION WRAPPER
    def generate_report(context_text, user_query, conflict_info):
        prompt = (
            f"### IDENTITY: Nexus 'Judge' Intelligence.\n"
            f"### INSTRUCTIONS: \n"
            f"1. Highlight any DISCREPANCIES between sources found by the Judge.\n"
            f"2. Compare Hard Law (EU) vs Soft Law (UN).\n"
            f"3. Maintain table structures and provide proof quotes.\n\n"
            f"### DISCREPANCY REPORT:\n{conflict_info}\n\n"
            f"### CONTEXT:\n{context_text}\n\n"
            f"### QUESTION:\n{user_query}\n\n"
            f"### EXECUTIVE REPORT:"
        )
        res = client_local.chat.completions.create(
            model="llama3", 
            messages=[{"role": "user", "content": prompt}],
            extra_body={"options": {"temperature": 0}}
        )
        return res.choices[0].message.content, count_tokens(prompt)

    answer, tokens_in = generate_report(full_context, query, discrepancy_report)
    stats['tokens_in'] += tokens_in

    # D. AGENTIC AUDIT
    audit_decision = agentic_audit(query, answer)
    if ("Article" in query or "%" in query) and "Article" not in answer:
        audit_decision = f"NEED: specific regulatory text for {query}"

    if "NEED:" in audit_decision:
        stats['agentic_pivots'] += 1
        sub_query = audit_decision.replace("NEED:", "").strip()
        print(f"🔄 AGENTIC PIVOT: Hunting for '{sub_query}'...")
        second_v = get_vector_search(sub_query, collection, 6)
        if second_v['documents'][0]:
            second_doc = "\n".join([f"[{m.get('source')}]: {d}" for d, m in zip(second_v['documents'][0], second_v['metadatas'][0])])
            answer, tokens_in_2 = generate_report(f"{full_context}\n\nADDITIONAL DISCOVERY:\n{second_doc}", query, discrepancy_report)
            stats['tokens_in'] += tokens_in_2

    # E. FINAL PACKAGING
    primary_meta = doc_to_meta.get(scored_results[0][0], {})
    evidence = {
        "files": stats['files_consulted'],
        "page": primary_meta.get("page", "N/A"),
        "snippet": scored_results[0][0][:250].replace("\n", " ") + "..." 
    }
    stats['tokens_out'] = count_tokens(answer)
    stats['total_ms'] = int((time.time() - start_total) * 1000)
    
    return answer, evidence, stats

# --- 5. THE UI ---
if __name__ == "__main__":
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*64)
        print("   N E X U S   'J U D G E'   I N T E L L I G E N C E   (V14)")
        print("="*64)
        
        user_query = input("\n🔎 ENTER RESEARCH QUERY (or 'exit'): ")
        if user_query.lower() in ['exit', 'quit']: break
        
        print("\n📡 Processing Signal (Conflict Detection Active)...")
        answer, evidence, stats = nexus_research_final(user_query)
        
        conf = min(100, max(0, int((stats['rerank_score'] + 5) * 18))) # Adjusted for -5.0 base

        print("\n" + "━"*64)
        print(f"📝 EXECUTIVE REPORT:\n{answer}")
        print("━"*64)
        
        if evidence:
            print(f"✅ TRUST VERIFICATION")
            print(f"   📍 SOURCES: {', '.join(evidence['files'])}")
            print(f"   🎯 CONFIDENCE: {conf}% | PIVOTS: {stats['agentic_pivots']}")
            
            with open("nexus_audit_log.md", "a", encoding="utf-8") as f:
                f.write(f"## 🛠️ SESSION ENTRY: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Query:** {user_query}\n")
                if stats['agentic_pivots'] > 0:
                    f.write(f"> 🔄 **AGENTIC EVENT:** Second-pass search triggered.\n")
                
                f.write(f"**Report:**\n{answer}\n\n")
                # FIXED: Logging multiple sources instead of evidence['file']
                f.write(f"**Verification:** Sources: {', '.join(evidence['files'])}, Page {evidence['page']} (Conf: {conf}%)\n")
                f.write(f"**Telemetry:** Total {stats['total_ms']}ms | In:{stats['tokens_in']} | Out:{stats['tokens_out']}\n")
                f.write(f"{'━'*40}\n\n")
        
        print(f"\n📊 TELEMETRY: {stats['total_ms']}ms | In:{stats['tokens_in']} | Out:{stats['tokens_out']}")
        print("="*64)
        input("\nPress Enter for next query...")