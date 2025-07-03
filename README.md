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

🚀 Usage
1. Download PubMed Data (Optional)
Before running the summarization tool, you can download and extract the required PubMed .xml.gz files by running:


# docker run summarization_tool python scripts/download_and_unzip_pubmed.py
# Note: By default, the script downloads the first 5 PubMed files listed in data/filelist.txt.
You can change this behavior by modifying the limit parameter in scripts/download_and_unzip_pubmed.py.

Make sure filelist.txt exists in the data/ directory with the list of filenames to download.
2. Run the Summarization Tool
Once the data is available in the data/ directory, run the summarization tool with your role and question:

docker run --env-file .env summarization_tool python main.py --role "pediatrician" --question "What are the latest treatments for juvenile arthritis?"
Make sure your .env file includes a valid OPENAI_API_KEY.


## 🐳 Docker

Build and run using:
```bash
docker build -t pubmed-summary .
docker run -it pubmed-summary

