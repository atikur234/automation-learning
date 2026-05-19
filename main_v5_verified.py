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

# print("🚀 Initializing Nexus Agentic Engine (Day 11)...")
print("🚀 Initializing Nexus Structural Engine (Day 12)...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')

def count_tokens(text):
    return len(tokenizer.encode(text))

# --- 2. SEARCH ENGINE MODULES ---
def get_vector_search(query, collection, n):
    return collection.query(query_texts=[query], n_results=n)

def get_keyword_search(query, all_docs, n):
    bm25 = BM25Okapi([doc.lower().split() for doc in all_docs])
    return bm25.get_top_n(query.lower().split(), all_docs, n=n)

# --- 3. DAY 11: AGENTIC AUDITOR (PRESERVED) ---
def agentic_audit(query, current_answer):
    """The 'Brain' that decides if we need to search deeper for context missing in the initial pass."""
    # audit_prompt = (
    #  f"### IDENTITY: You are a Senior Research Auditor.\n"
    #  f"### TASK: Analyze the answer for generic vs specific info. If funding amounts, specific donors, "
    #  f"or governance structures are mentioned in the query but missing in the answer, you MUST pivot.\n"
    #  f"### QUERY: {query}\n"
    #  f"### ANSWER: {current_answer}\n\n"
    #  f"INSTRUCTION: If details are vague, output 'NEED: [specific search term]'. Otherwise 'COMPLETE'."
    #  )
    # UPDATE: Day 13 Cross-Doc Auditor
    audit_prompt = (
       f"### IDENTITY: Senior Inter-Nexus Auditor.\n"
       f"### TASK: If the query asks to COMPARE two sources (e.g., UN and EU), "
       f"check if the answer has specific data from BOTH. If one side is vague or missing "
       f"Article/Section numbers, you MUST pivot.\n"
       f"### QUERY: {query}\n"
       f"### ANSWER: {current_answer}\n\n"
       f"INSTRUCTION: Output 'NEED: [source name] [topic]' or 'COMPLETE'."
     )
    
    response = client_local.chat.completions.create(
        model="llama3", 
        messages=[{"role": "user", "content": audit_prompt}],
        extra_body={"options": {"temperature": 0}}
    )
    return response.choices[0].message.content.strip()

# --- 3.5 DAY 12: RECURSIVE CHAPTER MAPPING (NEW) ---
# def get_recursive_summary(collection, page_number):
#     """
#     Zooms out to summarize the entire page context to find the 
#     'connective philosophy' between recommendations.
#     """
#     # Fetch all chunks from the specific page to get forest-level context
#     results = collection.get(where={"page": page_number})
#     if not results['documents']:
#         return "No broader chapter context found."
        
#     full_text = " ".join(results['documents'])
    
#     summary_prompt = (
#         f"### TASK: Summarize the underlying philosophy and narrative of this chapter section.\n"
#         f"### TEXT:\n{full_text[:4000]}\n\n"
#         f"### SUMMARY (Focus on the 'Why' and the connective tissue):"
#     )
    
#     response = client_local.chat.completions.create(
#         model="llama3", 
#         messages=[{"role": "user", "content": summary_prompt}],
#         extra_body={"options": {"temperature": 0}}
#     )
#     return response.choices[0].message.content
# --- 3.5 DAY 12: UPDATED CONTEXT WINDOWING ---
# def get_recursive_summary(collection, page_number):
#     # Fetch current, previous, and next page for a 3-page 'Forest View'
#     page_range = [page_number - 1, page_number, page_number + 1]
#     results = collection.get(where={"page": {"$in": page_range}})
    
#     if not results['documents']:
#         return "No broader chapter context found."
        
#     # Combine and summarize
#     full_text = " ".join(results['documents'])
#     # ... (rest of the summary logic)
     
#     summary_prompt = (
#         f"### TASK: Summarize the underlying philosophy and narrative of this chapter section.\n"
#         f"### TEXT:\n{full_text[:4000]}\n\n"
#         f"### SUMMARY (Focus on the 'Why' and the connective tissue):"
#     )
    
#     response = client_local.chat.completions.create(
#         model="llama3", 
#         messages=[{"role": "user", "content": summary_prompt}],
#         extra_body={"options": {"temperature": 0}}
#     )
#     return response.choices[0].message.content
# --- 3.5 DAY 12: UPDATED CONTEXT WINDOWING ---
def get_recursive_summary(collection, page_number):
    """
    Zooms out to summarize a 3-page window to capture the 
    narrative arc from problem to solution.
    """
    # Create a range: [Current-1, Current, Current+1]
    # This ensures we see what led up to the current section
    page_range = [max(1, page_number - 1), page_number, page_number + 1]
    
    # Updated ChromaDB query using the $in operator for metadata
    results = collection.get(where={"page": {"$in": page_range}})
    
    if not results['documents']:
        return "No broader chapter context found."
        
    full_text = " ".join(results['documents'])
    
    summary_prompt = (
        f"### TASK: Analyze the NARRATIVE ARC of these pages.\n"
        f"### TEXT:\n{full_text[:6000]}\n\n"
        f"### SUMMARY (Focus on the 'Problem -> Solution' logic):"
    )
    
    response = client_local.chat.completions.create(
        model="llama3", 
        messages=[{"role": "user", "content": summary_prompt}],
        extra_body={"options": {"temperature": 0}}
    )
    return response.choices[0].message.content

# --- 4. THE EXECUTIVE RESEARCH ENGINE ---
def nexus_research_final(query):
    # Added 'files_consulted' to stats for multi-doc tracking
    stats = {'retrieval_ms': 0, 'rerank_ms': 0, 'gen_ms': 0, 'total_ms': 0, 'tokens_in': 0, 'tokens_out': 0, 'rerank_score': 0, 'agentic_pivots': 0, 'recursive_summaries': 0, 'files_consulted': []}
    start_total = time.time()
    
    client_db = chromadb.PersistentClient(path="./chroma_db")
    collection = client_db.get_collection(name="nexus_compliance_vault")
    all_data = collection.get()

    # A. INITIAL RETRIEVAL & RERANK
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Day 13: Increased N to 8 to catch both documents in the first net
        f_v = executor.submit(get_vector_search, query, collection, 8) 
        f_k = executor.submit(get_keyword_search, query, all_data['documents'], 8)
        v_docs = f_v.result()['documents'][0]
        k_docs = f_k.result()
    
    candidates = list(set(v_docs + k_docs))
    doc_to_meta = {doc: meta for doc, meta in zip(all_data['documents'], all_data['metadatas'])}
    
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    scored_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    # PATCH 1: Lowered threshold to -5.0 to catch technical regulatory text
    if not scored_results or scored_results[0][1] < -5.0:
        return "❌ Data missing in vault. Check if EU AI Act is properly ingested.", None, stats

    best_score = scored_results[0][1]
    stats['rerank_score'] = best_score

    # --- DAY 13: SOURCE ROUTING ---
    context_blocks = []
    seen_sources = set()
    
    # Process top 6 results to ensure multi-source representation
    for doc, score in scored_results[:6]:
        meta = doc_to_meta.get(doc, {})
        source_name = meta.get("source", "Unknown")
        seen_sources.add(source_name)
        context_blocks.append(f"--- SOURCE: {source_name} (Page {meta.get('page', 'N/A')}) ---\n{doc}")

    stats['files_consulted'] = list(seen_sources)
    best_doc = "\n\n".join(context_blocks)

    # --- DAY 12: STRUCTURAL DETECTION & RECURSION (PRESERVED) ---
    primary_meta = doc_to_meta.get(scored_results[0][0], {})
    broad_terms = ["philosophy", "vision", "connect", "underlying", "overall", "narrative", "compare"]
    is_broad = any(term in query.lower() for term in broad_terms)
    
    if is_broad:
        print(f"🌲 FOREST VIEW: Analyzing {primary_meta.get('source')} narrative...")
        stats['recursive_summaries'] += 1
        page_ref = primary_meta.get("page", 1)
        # Day 13: Passing source name to ensure summary stays in the correct document
        chapter_philosophy = get_recursive_summary(collection, page_ref, primary_meta.get("source"))
        best_doc = f"CHAPTER PHILOSOPHY ({primary_meta.get('source')}):\n{chapter_philosophy}\n\n{best_doc}"

    # --- B. GENERATION WRAPPER ---
    def generate_report(context_text, user_query):
        prompt = (
            f"### IDENTITY: You are Nexus Inter-Nexus Intelligence.\n"
            f"### INSTRUCTIONS: \n"
            f"1. Compare data across SOURCES (e.g., UN Report vs EU AI Act).\n"
            f"2. Cite specific Article numbers and turnover percentages (if found).\n"
            f"3. Note if one document is voluntary and the other is mandatory.\n\n"
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

    answer, tokens_in = generate_report(best_doc, query)
    stats['tokens_in'] += tokens_in

    # --- C. DAY 11/13 AGENTIC AUDIT & SECOND PASS ---
    audit_decision = agentic_audit(query, answer)
    
    # PATCH 2: Hard-Override Pivot for missing "Articles" or "Percentages"
    if ("Article" in query or "%" in query) and "Article" not in answer:
        audit_decision = f"NEED: specific regulatory text for {query}"

    if "NEED:" in audit_decision:
        stats['agentic_pivots'] += 1
        sub_query = audit_decision.replace("NEED:", "").strip()
        print(f"🔄 AGENTIC PIVOT: Context insufficient. Searching across vault for '{sub_query}'...")
        
        # Day 13: Increase N to 6 for the second pass to look deeper
        second_v = get_vector_search(sub_query, collection, 6)
        if second_v['documents'][0]:
            # Interleave sources for the secondary context
            second_doc = "\n".join([f"[{m.get('source')}]: {d}" for d, m in zip(second_v['documents'][0], second_v['metadatas'][0])])
            combined_context = f"PRIMARY DATA:\n{best_doc}\n\nADDITIONAL DISCOVERY:\n{second_doc}"
            answer, tokens_in_2 = generate_report(combined_context, query)
            stats['tokens_in'] += tokens_in_2

    # D. FINAL PACKAGING
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
        print("   N E X U S   R E S E A R C H   I N T E L L I G E N C E   (V12)")
        print("="*64)
        
        user_query = input("\n🔎 ENTER RESEARCH QUERY (or 'exit'): ")
        if user_query.lower() in ['exit', 'quit']: break
        
        print("\n📡 Processing Signal (Structural Mode Active)...")
        answer, evidence, stats = nexus_research_final(user_query)
        
        conf = min(100, max(0, int((stats['rerank_score'] + 4) * 20))) 

        print("\n" + "━"*64)
        print(f"📝 EXECUTIVE REPORT:\n{answer}")
        print("━"*64)
        
        # In your __main__ loop:
        if evidence:
            print(f"✅ TRUST VERIFICATION")
            print(f"   📍 SOURCES: {', '.join(evidence['files'])}") # Changed to join list
            print(f"   🎯 CONFIDENCE: {conf}% | PIVOTS: {stats['agentic_pivots']}")
            
            with open("nexus_audit_log.md", "a", encoding="utf-8") as f:
                f.write(f"## 🛠️ SESSION ENTRY: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Query:** {user_query}\n")
                if stats['recursive_summaries'] > 0:
                    f.write(f"> 🌲 **FOREST VIEW:** Recursive chapter summarization performed.\n")
                if stats['agentic_pivots'] > 0:
                    f.write(f"> 🔄 **AGENTIC EVENT:** System detected context gaps and performed a second-pass search.\n")
                
                f.write(f"**Report:**\n{answer}\n\n")
                f.write(f"**Verification:** Source {evidence['file']}, Page {evidence['page']} (Conf: {conf}%)\n")
                f.write(f"**Telemetry:** Total {stats['total_ms']}ms | In:{stats['tokens_in']} | Out:{stats['tokens_out']}\n")
                f.write(f"{'━'*40}\n\n")
        
        print(f"\n📊 TELEMETRY: {stats['total_ms']}ms | In:{stats['tokens_in']} | Out:{stats['tokens_out']}")
        print("="*64)
        input("\nPress Enter for next query...")