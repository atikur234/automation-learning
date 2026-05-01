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

print("🚀 Initializing Nexus Agentic Engine (Day 11)...")
# Re-enabling GPU for the RTX 5050 Blackwell Architecture
# reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda') 
# Back to CPU for stability during environment sync
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')

def count_tokens(text):
    return len(tokenizer.encode(text))

# --- 2. SEARCH ENGINE MODULES ---
def get_vector_search(query, collection, n):
    return collection.query(query_texts=[query], n_results=n)

def get_keyword_search(query, all_docs, n):
    bm25 = BM25Okapi([doc.lower().split() for doc in all_docs])
    return bm25.get_top_n(query.lower().split(), all_docs, n=n)

# --- 3. DAY 11: AGENTIC AUDITOR (NEW) ---
def agentic_audit(query, current_answer):
    """The 'Brain' that decides if we need to search deeper for context missing in the initial pass."""
    # audit_prompt = (
    #     f"### IDENTITY: You are a Research Auditor.\n"
    #     f"### TASK: Review the answer to see if it mentions missing details like 'Annexes', 'Tables', or 'further sections'.\n"
    #     f"### QUERY: {query}\n"
    #     f"### ANSWER: {current_answer}\n\n"
    #     f"INSTRUCTION: If the answer suggests more info is needed elsewhere, output 'NEED: [specific topic to search]'. "
    #     f"Otherwise, output 'COMPLETE'."
    # )
    # UPDATE: Strengthening the Auditor's 'Skepticism'
    audit_prompt = (
     f"### IDENTITY: You are a Senior Research Auditor.\n"
     f"### TASK: Analyze the answer for generic vs specific info. If funding amounts, specific donors, "
     f"or governance structures are mentioned in the query but missing in the answer, you MUST pivot.\n"
     f"### QUERY: {query}\n"
     f"### ANSWER: {current_answer}\n\n"
     f"INSTRUCTION: If details are vague, output 'NEED: [specific search term]'. Otherwise 'COMPLETE'."
     )
    
    response = client_local.chat.completions.create(
        model="llama3", 
        messages=[{"role": "user", "content": audit_prompt}],
        extra_body={"options": {"temperature": 0}}
    )
    return response.choices[0].message.content.strip()

# --- 4. THE EXECUTIVE RESEARCH ENGINE ---
def nexus_research_final(query):
    stats = {'retrieval_ms': 0, 'rerank_ms': 0, 'gen_ms': 0, 'total_ms': 0, 'tokens_in': 0, 'tokens_out': 0, 'rerank_score': 0, 'agentic_pivots': 0}
    start_total = time.time()
    
    client_db = chromadb.PersistentClient(path="./chroma_db")
    collection = client_db.get_collection(name="nexus_compliance_vault")
    all_data = collection.get()

    # A. INITIAL RETRIEVAL & RERANK
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_v = executor.submit(get_vector_search, query, collection, 5)
        f_k = executor.submit(get_keyword_search, query, all_data['documents'], 5)
        v_docs = f_v.result()['documents'][0]
        k_docs = f_k.result()
    
    candidates = list(set(v_docs + k_docs))
    doc_to_meta = {doc: meta for doc, meta in zip(all_data['documents'], all_data['metadatas'])}
    
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    scored_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    if not scored_results or scored_results[0][1] < -4.0:
        return "❌ Data missing in vault.", None, stats

    # --- DAY 10 SYNTHESIS LOGIC (PRESERVED) ---
    best_score = scored_results[0][1]
    stats['rerank_score'] = best_score
    
    comparison_keywords = ["compare", "difference", "versus", "both", "opportunity scan"]
    if any(k in query.lower() for k in comparison_keywords) and len(scored_results) > 1:
        best_doc = f"--- CONTEXT A ---\n{scored_results[0][0]}\n\n--- CONTEXT B ---\n{scored_results[1][0]}"
    else:
        best_doc = scored_results[0][0]

    # --- B. DAY 11 GENERATION WRAPPER ---
    def generate_report(context_text, user_query):
        prompt = (
            f"### IDENTITY: You are Nexus Research Intelligence. Provide a precise executive report.\n"
            f"### INSTRUCTIONS: \n"
            f"1. Summarize key facts in bullet points.\n"
            f"2. Compare specific sections if multiple contexts are provided.\n"
            f"3. Maintain table structures and provide a proof quote.\n\n"
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

    # REPLACED: Old Single Pass Logic (Commented Out)
    # response = client_local.chat.completions.create(...)
    # answer = response.choices[0].message.content
    
    # NEW: First Pass Generation
    answer, tokens_in = generate_report(best_doc, query)
    stats['tokens_in'] += tokens_in

    # --- C. DAY 11 AGENTIC AUDIT & SECOND PASS ---
    audit_decision = agentic_audit(query, answer)
    
    if "NEED:" in audit_decision:
        stats['agentic_pivots'] += 1
        sub_query = audit_decision.replace("NEED:", "").strip()
        print(f"🔄 AGENTIC PIVOT: Context insufficient. Searching for '{sub_query}'...")
        
        # Trigger second retrieval pass
        second_v = get_vector_search(sub_query, collection, 2)
        if second_v['documents'][0]:
            second_doc = second_v['documents'][0][0]
            
            # Synthesis: Combine original and new discovery
            combined_context = f"PRIMARY DATA:\n{best_doc}\n\nADDITIONAL DISCOVERY:\n{second_doc}"
            answer, tokens_in_2 = generate_report(combined_context, query)
            stats['tokens_in'] += tokens_in_2

    # D. FINAL PACKAGING & METADATA
    primary_meta = doc_to_meta.get(scored_results[0][0], {})
    evidence = {
        "file": primary_meta.get("source", "Unknown"),
        "page": primary_meta.get("page", "N/A"),
        "snippet": scored_results[0][0][:250].replace("\n", " ") + "..." 
    }
    
    stats['tokens_out'] = count_tokens(answer)
    stats['total_ms'] = int((time.time() - start_total) * 1000)
    
    return answer, evidence, stats

# --- 5. THE UI (UPDATED FOR AGENTIC FEEDBACK) ---
# if __name__ == "__main__":
#     while True:
#         os.system('cls' if os.name == 'nt' else 'clear')
#         print("="*64)
#         print("   N E X U S   R E S E A R C H   I N T E L L I G E N C E   (V11)")
#         print("="*64)
        
#         user_query = input("\n🔎 ENTER RESEARCH QUERY (or 'exit'): ")
#         if user_query.lower() in ['exit', 'quit']: break
        
#         print("\n📡 Processing Signal (Agentic Mode Active)...")
#         answer, evidence, stats = nexus_research_final(user_query)
        
#         conf = min(100, max(0, int((stats['rerank_score'] + 4) * 20))) 
#         if conf < 30:
#              print("⚠️  WARNING: High uncertainty. The vault may not contain specific quantitative data for this query.")

#         print("\n" + "━"*64)
#         print(f"📝 EXECUTIVE REPORT:\n{answer}")
#         print("━"*64)
        
#         # --- DAY 11: EXPANDED AUDIT LOGGING ---
#         if evidence:
#             # - Trust Verification logic
#             print(f"✅ TRUST VERIFICATION")
#             print(f"   📍 SOURCE: {evidence['file']} | PAGE: {evidence['page']}")
#             print(f"   🎯 CONFIDENCE: {conf}% | PIVOTS: {stats['agentic_pivots']}")
            
#             with open("nexus_audit_log.md", "a", encoding="utf-8") as f:
#                 f.write(f"## 🛠️ SESSION ENTRY: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
#                 f.write(f"**Query:** {user_query}\n")
                
#                 # Log if a pivot happened
#                 if stats['agentic_pivots'] > 0:
#                     f.write(f"> 🔄 **AGENTIC EVENT:** System detected context gaps and performed a second-pass search.\n")
                
#                 f.write(f"**Report:**\n{answer}\n\n")
#                 f.write(f"**Verification:** Source {evidence['file']}, Page {evidence['page']} (Conf: {conf}%)\n")
#                 f.write(f"**Telemetry:** Total {stats['total_ms']}ms | In:{stats['tokens_in']} | Out:{stats['tokens_out']}\n")
#                 f.write(f"{'━'*40}\n\n")
# --- 5. THE UI (UPDATED FOR AUDIT LOGGING) ---
if __name__ == "__main__":
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*64)
        print("   N E X U S   R E S E A R C H   I N T E L L I G E N C E   (V11)")
        print("="*64)
        
        user_query = input("\n🔎 ENTER RESEARCH QUERY (or 'exit'): ")
        if user_query.lower() in ['exit', 'quit']: break
        
        print("\n📡 Processing Signal (Agentic Mode Active)...")
        answer, evidence, stats = nexus_research_final(user_query)
        
        conf = min(100, max(0, int((stats['rerank_score'] + 4) * 20))) 

        print("\n" + "━"*64)
        print(f"📝 EXECUTIVE REPORT:\n{answer}")
        print("━"*64)
        
        if evidence:
            print(f"✅ TRUST VERIFICATION")
            print(f"   📍 SOURCE: {evidence['file']} | PAGE: {evidence['page']}")
            print(f"   🎯 CONFIDENCE: {conf}% | PIVOTS: {stats['agentic_pivots']}")
            print(f"   📜 PROOF: \"{evidence['snippet']}\"")
            
            # --- DAY 11: EXPANDED AUDIT LOGGING ---
            with open("nexus_audit_log.md", "a", encoding="utf-8") as f:
                f.write(f"## 🛠️ SESSION ENTRY: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Query:** {user_query}\n")
                
                # Check if the Agent triggered a second pass
                if stats.get('agentic_pivots', 0) > 0:
                    f.write(f"> 🔄 **AGENTIC EVENT:** System detected context gaps and performed a second-pass search.\n")
                else:
                    f.write(f"> ✅ **SINGLE PASS:** Initial search provided sufficient context.\n")
                
                f.write(f"**Report:**\n{answer}\n\n")
                f.write(f"**Verification:** Source {evidence['file']}, Page {evidence['page']} (Conf: {conf}%)\n")
                f.write(f"**Telemetry:** Total {stats['total_ms']}ms | In:{stats['tokens_in']} | Out:{stats['tokens_out']}\n")
                f.write(f"{'━'*40}\n\n")
        
        print(f"\n📊 TELEMETRY: {stats['total_ms']}ms | In:{stats['tokens_in']} | Out:{stats['tokens_out']}")
        print("="*64)
        input("\nPress Enter for next query...")