import os
import time
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi # New: Keyword Search
from sentence_transformers import CrossEncoder # New: Precision Reranker

client_local = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

# Load the Reranker (This will download once ~200MB)
# This 'ms-marco' model is the gold standard for reranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def get_hybrid_context(query, collection, n_initial=15): # Increased n_initial for better recall
    # 1. Broad Net (Same as before)
    vector_results = collection.query(query_texts=[query], n_results=n_initial)
    vector_docs = vector_results['documents'][0]
    vector_metas = vector_results['metadatas'][0]
    
    # 2. Hybrid Retrieval (BM25)
    all_data = collection.get()
    all_docs = all_data['documents']
    all_metas = all_data['metadatas']
    
    bm25 = BM25Okapi([doc.lower().split() for doc in all_docs])
    keyword_docs = bm25.get_top_n(query.lower().split(), all_docs, n=n_initial)
    
    # 3. CONTEXTUAL INJECTION (The Fix)
    # We combine doc text with its source/section info for the Reranker to 'see'
    candidate_docs = list(set(vector_docs + keyword_docs))
    
    # We create a mapping of doc to its metadata for quick lookup
    doc_to_meta = {doc: meta for doc, meta in zip(all_docs, all_metas)}
    
    # Enhanced pairs for the Reranker
    # We add the source name or chunk index to the text to give it 'scoping'
    enriched_pairs = []
    for doc in candidate_docs:
        meta = doc_to_meta.get(doc, {})
        source = meta.get('source', 'Unknown')
        # We 'wrap' the text to tell the Reranker exactly what this is
        enriched_text = f"SOURCE: {source} | CONTENT: {doc}"
        enriched_pairs.append([query, enriched_text])
    
    # 4. RERANKING
    scores = reranker.predict(enriched_pairs)
    scored_docs = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs[:3]]

def ask_nexus_v3(query):
    client_db = chromadb.PersistentClient(path="./chroma_db")
    collection = client_db.get_collection(name="nexus_compliance_vault")
    
    # Stage 1 & 2: Hybrid + Rerank
    best_chunks = get_hybrid_context(query, collection)
    context = "\n\n".join(best_chunks)
    
    # Stage 3: Generate
    prompt = f"Using ONLY this context, answer: {query}\n\nCONTEXT:\n{context}"
    response = client_local.chat.completions.create(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # TEST: The "Specific Entity" Needle
    print(ask_nexus_v3("Compare the representation of countries in AI initiatives between the UN report and the Pulse Check survey."))