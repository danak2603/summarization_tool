# Summarization Tool

This tool extracts relevant biomedical literature from PubMed and PMC, and generates concise summaries tailored to specific user roles using an LLM.

## 📦 Project Structure

├── app/
│ ├── init.py
│ ├── summarizer.py
│ ├── retriever.py
│ └── ...
├── data/
│ └── (leave empty, used by the download script)
├── main.py
├── Dockerfile
├── requirements.txt
└── README.md

# ✍ Inputs
User role (e.g., rheumatologist, biomedical researcher)

Research question (e.g., "What are the latest treatment options for juvenile arthritis?")

# 📤 Output
Concise summary (≤ 5,000 characters)

Citations in [PMID...] or [PMC...] format


# 🚀 How to Run

1. Clone the repository
2. Run the download script to fetch data (based on `filelist.txt`)
3. Use the CLI or Docker to run the summarization tool


## 🐳 Docker

Build and run using:
```bash
docker build -t pubmed-summary .
docker run -it pubmed-summary

