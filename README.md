# 🛰️ Nexus Intelligence Director (v15.0)
**Agentic Multi-Doc RAG for Regulatory & Policy Intelligence**

Nexus is a production-ready, localized AI agent designed to bridge the gap between **"Hard Law"** (e.g., EU AI Act) and **"Soft Law"** (e.g., UN AI Reports). Unlike standard RAG systems that simply retrieve text, Nexus performs Structural Auditing, Recursive Summarization, and Conflict Detection to provide decision-makers with a strategic executive dashboard.

---

## 🚀 System Architecture

Nexus operates through a four-phase Agentic Pipeline:

1. **Structural Ingestion:**  
   Uses `pdfplumber` and `SemanticChunker` to preserve document hierarchy, table structures, and page metadata.

2. **The Judge Protocol:**  
   A specialized pass that identifies direct contradictions between retrieved sources.

3. **Agentic Pivot:**  
   If the initial search fails to find specific regulatory data (like Article numbers or fine percentages), the system triggers a secondary "Deep Search."

4. **Triple-Lens Synthesis:**  
   The final output is rendered through a Regulatory, Diplomatic, and Risk-based lens.

---

## 🛠️ Tech Stack

- **LLM:** Llama 3 (via Ollama)
- **Vector DB:** ChromaDB
- **Embedding Model:** all-MiniLM-L6-v2
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2
- **Parsing:** pdfplumber
- **Tokenizer:** tiktoken

---

## 📂 Project Structure

```
├── data/                    # Drop your PDFs here (UN, EU, etc.)
├── chroma_db/               # Persistent Vector Store
├── nexus_ingestor.py        # Multi-doc structural ingestion script
├── nexus_engine.py          # Main Intelligence Director script
├── nexus_audit_log.md       # Automated session history & telemetry
└── requirements.txt         # Project dependencies
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed

Download the Llama 3 model:
```bash
ollama run llama3
```

### 2. Install Dependencies

```bash
pip install chromadb langchain-huggingface langchain-experimental sentence-transformers pdfplumber rank_bm25 tiktoken openai
```

### 3. Ingest Your Documents

Place your PDFs (e.g., `un_ai_report.pdf`, `eu_ai_act.pdf`) into the `/data` folder, then run:

```bash
python nexus_ingestor.py
```

### 4. Run the Intelligence Director

```bash
python nexus_engine.py
```

---

## ⚖️ Core Features

### 🌲 Forest View (Recursive Logic)
For broad queries (e.g., "What is the vision?"), the system "zooms out" to summarize entire chapters before drilling down into specific details—ensuring context isn't lost in small snippets.

### ⚖️ The Judge Pass
Nexus cross-references sources. If the EU mandates a ban while the UN remains voluntary, the system generates a Discrepancy Report to highlight legal tension.

### 🔄 Agentic Pivot
The system audits its own answers. Upon detecting a "Data Void" (e.g., missing specific legal articles), it automatically pauses, re-queries the vault with high-precision terms, and regenerates the brief.

### 📊 Telemetry & Audit
Every session is recorded in `nexus_audit_log.md`, including:

- **Sources Consulted:** List of specific documents used.
- **Confidence Score:** Normalized reranker scores.
- **Agentic Pivots:** Number of times the AI felt "insufficiently informed."
- **Tokens & Latency:** Computational efficiency tracking.

---

## 🧠 Strategic Value

Nexus enables legal, compliance, and policy teams to surface direct conflicts and hidden synergies across global regulatory frameworks, providing actionable intelligence rather than mere text retrieval.

---

## License

[MIT License](LICENSE) (or specify your license)

---

*For questions or support, please contact the project maintainer.*
