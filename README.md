# Summarization Tool

This repository contains a command-line tool for generating concise biomedical summaries tailored to specific user roles and research questions. It retrieves scientific articles, builds a context-aware prompt, generates a summary using OpenAI models, and evaluates it using automated metrics.

---

## Project Structure

```
.
├── app/
│   ├── data_loader.py        # Functions for parsing and filtering PubMed/PMC data
│   ├── retrieval.py          # Topic extraction and document retrieval logic
│   ├── summarizer.py         # Prompt creation and summary generation
│   ├── evaluator.py          # LLM-based evaluation of summaries
│   └── kpis.py               # KPI computations (e.g., similarity, citation count)
├── data/                     # Folder to store downloaded XML files
├── download_and_unzip_pubmed.py  # Script to download and extract article files
├── Dockerfile
├── main.py                  # Entry point for running the summarization tool
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/danak2603/summarization_tool.git
cd summarization_tool
```

### 2. Prepare the Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add OpenAI API Key

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Download the Data

Use the provided script to download PubMed and PMC data into the `data/` directory:

```bash
python download_and_unzip_pubmed.py --filelist filelist.txt
```

Ensure the data is stored in:

```
data/
├── pubmed24n0001.xml
├── pmc_comm_use_subset/
│   ├── ...
```

---

## Run the Tool with Docker

### 1. Make Sure the `.env` File Exists

```env
OPENAI_API_KEY=your_openai_api_key_here
```
### 2. Build the Docker Image

#### Before running the tool, you need to build the Docker image:

```bash
docker build -t summarization_tool .
```

### 3. Use the Appropriate Docker Command

#### On **Mac/Linux**:

```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  summarization_tool \
  python main.py --role "pediatrician" --question "What are the latest treatment options for juvenile arthritis?"
```

#### On **Windows PowerShell**:

```powershell
docker run --rm `
  --env-file .env `
  -v "${PWD}/data:/app/data" `
  summarization_tool `
  python main.py --role "pediatrician" --question "What are the latest treatment options for juvenile arthritis?"
```

---

## Run Without Docker (Optional)

```bash
python main.py --role "pediatrician" --question "What are the latest treatment options for juvenile arthritis?"
```

---
## How the Tool Works

1. Extracts key topics from the input question  
2. Expands the topics for better document filtering  
3. Parses PubMed and PMC articles  
4. Retrieves the most relevant documents using a FAISS vector store  
5. Generates a summary using an LLM (via OpenAI API)  
6. Evaluates the summary and computes KPIs


## Output

The tool prints:

- The generated summary
- An evaluation report (LLM-based rubric scoring)
- KPI metrics (e.g., token count, citation count, semantic similarity)

---

## Notes

- Supported user roles include: "pediatrician", "general practitioner", "researcher", etc.
- The summary is tailored to the user role: e.g., simplified language for GPs, technical details for researchers.
- No data artifacts are included in the repository, in accordance with the submission guidelines.
- Ensure that the data is downloaded using the provided script before running the summarization tool.

---

## Customization

You may modify the user role, question directly in `main.py` or via the CLI.

