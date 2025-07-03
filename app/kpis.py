from typing import Dict
import re
from typing import List
import numpy as np
from config import OPENAI_API_KEY, client


def count_citations(summary: str) -> int:
    """
    Count citations from PubMed and PMC in the summary.
    Looks for [PMID12345] and [PMC12345] patterns.
    """
    return len(re.findall(r"\[PM(C|ID)\d+\]", summary))

def count_tokens(summary: str, model_name: str = "gpt-4") -> int:
    """
    Estimate the number of tokens used in the summary using OpenAI tokenizer.
    """
    messages = [
        {"role": "system", "content": "Count the number of tokens in the following text."},
        {"role": "user", "content": summary},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=1
    )
    return response.usage.total_tokens

def count_source_documents(documents: List[dict]) -> int:
    """
    Count how many documents were used to generate the summary.
    """
    return len(documents)

def compute_avg_llm_score(evaluation_report: Dict[str, dict]) -> float:
    """
    Compute average score from the evaluation report.
    """
    return round(
        sum([v["score"] for v in evaluation_report.values()]) / len(evaluation_report),
        2
    )

def compute_semantic_similarity_to_query(summary: str, query: str, model_name: str = "text-embedding-3-small") -> float:
    """
    Compute cosine similarity between the query and generated summary using OpenAI embeddings.
    Returns a float between 0 (unrelated) and 1 (identical).
    """
    embeddings = client.embeddings.create(
        model=model_name,
        input=[query, summary]
    )
    vec_query = np.array(embeddings.data[0].embedding)
    vec_summary = np.array(embeddings.data[1].embedding)

    cosine_sim = np.dot(vec_query, vec_summary) / (np.linalg.norm(vec_query) * np.linalg.norm(vec_summary))
    return round(float(cosine_sim), 4)

