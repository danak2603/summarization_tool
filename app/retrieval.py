from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from openai import OpenAI
from config import OPENAI_API_KEY, client

def step_back_and_extract_topics(question, model="gpt-3.5-turbo"):
    prompt = f"""
    You are a biomedical research assistant.

    Your job is to analyze a user's research question and write a *step-back summary* — a 1–2 sentence explanation that captures the broader 
    biomedical context of the question.

    The summary should NOT rephrase the question.
    Instead, it should generalize it to reflect the underlying medical domain, disease area, or biological system it relates to.

    Then, return a list of relevant biomedical topics (specific + general) that can help retrieve scientific articles related to the question.

    Format:
    User Question: [original question]  
    Step-Back Summary: [general explanation of what this question is about]  
    Topics: [list of 3–6 phrases, both specific and general]

    Examples:

    User Question: What are the known side effects of infliximab in long-term use?
    Step-Back Summary: This question is about safety and long-term adverse effects of infliximab, a monoclonal antibody used to treat 
    autoimmune diseases such as Crohn's disease and rheumatoid arthritis.  
    Topics: ["infliximab", "autoimmune diseases", "Crohn's disease", "rheumatoid arthritis", "drug safety"]

    User Question: What are the comparative effects of etanercept and adalimumab in treating psoriatic arthritis?
    Step-Back Summary: This question relates to the evaluation of biologic therapies used to treat psoriatic arthritis, a type of autoimmune 
    inflammatory arthritis associated with psoriasis.
    Topics: ["etanercept", "adalimumab", "psoriatic arthritis", "biologic therapies", "autoimmune diseases", "inflammatory arthritis"]

    ---

    User Question: {question}
    Step-Back Summary:
    Topics:
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    output = response.choices[0].message.content

    lines = output.strip().splitlines()
    summary_line = next((line for line in lines if line.startswith("Step-Back Summary:")), "")
    topics_line = next((line for line in lines if line.startswith("Topics:")), "")

    summary = summary_line.replace("Step-Back Summary:", "").strip()
    try:
        topics = eval(topics_line.replace("Topics:", "").strip())
    except:
        topics = []

    return summary, topics


def softly_expand_topics(topics, model="gpt-3.5-turbo"):
    prompt = f"""
    You are a biomedical assistant.

    For each of the following biomedical search topics, suggest 1–2 broader medical terms that can improve retrieval in article search.
    Only include terms that are *truly broader* — not just related, narrower, or synonymous.
    Avoid overly general terms such as "treatment", "therapy", "biologics", or "medicine".
    Return a valid Python list of the expanded set only (no explanations, no formatting).

    Original topics:
    {topics}

    Expanded topics:
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    output = response.choices[0].message.content.strip()

    try:
        expanded = eval(output)
    except:
        expanded = []

    return list(set(topics) | set(expanded))


def build_faiss_vectorstore(documents):
    embedding_model = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embedding_model)
    return vectorstore
