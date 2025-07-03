import argparse
import json
from app.data_loader import load_and_prepare_documents
from app.retrieval import (
    step_back_and_extract_topics,
    softly_expand_topics,
    build_faiss_vectorstore,
)
from app.summarizer import generate_summary_from_documents
from app.evaluator import evaluate_summary
from app.kpis import (compute_avg_llm_score,
                      count_citations,
                      count_tokens,
                      count_source_documents,
                      compute_semantic_similarity_to_query)


def generate_summary(user_role: str, user_question: str,pmc_limit: int = None):
    """
        Main pipeline to generate a biomedical summary based on user role and question.

        Steps:
        - Extracts topics from the question
        - Expands topics for better filtering
        - Loads and prepares articles from PubMed/PMC
        - Retrieves relevant documents using FAISS
        - Generates a summary and evaluates it using LLM
        - Computes relevant KPIs
    """

    step_back_summary, topics = step_back_and_extract_topics(user_question)
    expand_topics = softly_expand_topics(topics)

    all_docs = load_and_prepare_documents(expand_topics,pmc_limit=pmc_limit)
    vectorstore = build_faiss_vectorstore(all_docs)

    query_text = f"""Question: {user_question}
    General Context: {step_back_summary}""".strip()
    similar_docs = vectorstore.similarity_search(query_text, k=7)

    summary = generate_summary_from_documents(user_role, user_question, similar_docs)
    evaluation_report = evaluate_summary(user_role, user_question, summary)

    kpis = {
        "avg_llm_score": compute_avg_llm_score(evaluation_report),
        "num_citations": count_citations(summary),
        "num_tokens": count_tokens(summary),
        "num_source_documents": count_source_documents(similar_docs),
        "semantic_similarity_to_query": compute_semantic_similarity_to_query(summary, query_text)
    }
    return summary, evaluation_report, kpis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate biomedical summaries from scientific articles.")
    parser.add_argument("--role", required=True, help="User role, e.g., 'pediatrician'")
    parser.add_argument("--question", required=True, help="Research question to answer")
    parser.add_argument("--pmc_limit", type=int, default=None, help="Optional limit on number of PMC files to load")

    args = parser.parse_args()

    summary_result, evaluation_report_result, kpis_result  = generate_summary(user_role=args.role,
                                                                              user_question=args.question,
                                                                              pmc_limit=args.pmc_limit)

    print('summary:',summary_result)
    print('\n\n')
    print("Evaluation Report:")
    print(json.dumps(evaluation_report_result, indent=2, ensure_ascii=False))
    print("\nKPIs:")
    print(json.dumps(kpis_result, indent=2, ensure_ascii=False))
