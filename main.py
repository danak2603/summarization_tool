import argparse
import glob
import os

from app.data_loader import (
    parse_pubmed_file_filtered,
    filter_pubmed_articles_by_topics,
    parse_folder_pmc,
    filter_pmc_articles_by_topics,
    prepare_pubmed_documents,
    prepare_pmc_documents,
)
from app.retrieval import (
    step_back_and_extract_topics,
    softly_expand_topics,
    build_faiss_vectorstore,
)
from app.summarizer import generate_summary_from_documents


def generate_summary(user_role: str, user_question: str) -> str:
    step_back_summary, topics = step_back_and_extract_topics(user_question)
    expand_topics = softly_expand_topics(topics)
    print(f"Expanded Topics: {expand_topics}")

    pubmed_files = glob.glob("data/pubmed*.xml")
    parsed_articles_pubmed = []
    for file_path in pubmed_files:
        parsed_articles_pubmed.extend(parse_pubmed_file_filtered(file_path))
    filtered_articles_pubmed = filter_pubmed_articles_by_topics(parsed_articles_pubmed, expand_topics)

    pmc_dirs = [os.path.join("data", d) for d in os.listdir("data")
                if d.lower().startswith("pmc") and os.path.isdir(os.path.join("data", d))]
    articles_pmc = []
    for folder_path in pmc_dirs:
        articles_pmc.extend(parse_folder_pmc(folder_path, include_body=True, limit=500))
    filtered_articles_pmc = filter_pmc_articles_by_topics(articles_pmc, expand_topics, include_body_in_filter=True)

    pubmed_docs = prepare_pubmed_documents(filtered_articles_pubmed)
    pmc_docs = prepare_pmc_documents(filtered_articles_pmc)
    all_docs = pubmed_docs + pmc_docs

    vectorstore = build_faiss_vectorstore(all_docs)

    query_text = f"""Question: {user_question}
General Context: {step_back_summary}""".strip()
    similar_docs = vectorstore.similarity_search(query_text, k=7)

    return generate_summary_from_documents(user_role, user_question, similar_docs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate biomedical summaries from scientific articles.")
    parser.add_argument("--role", required=True, help="User role, e.g., 'pediatrician'")
    parser.add_argument("--question", required=True, help="Research question to answer")

    args = parser.parse_args()

    summary = generate_summary(user_role=args.role, user_question=args.question)

    print("\n--- Summary ---\n")
    print(summary)
