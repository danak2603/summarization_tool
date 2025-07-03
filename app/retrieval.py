from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY, client

def step_back_and_extract_topics(question, model="gpt-3.5-turbo"):
    """
    Extracts a broader biomedical context and related topics from a user question to support document retrieval.
    """
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


def softly_expand_topics(topics, model="gpt-3.5-turbo", max_terms=15):
    """
    Expands a list of biomedical topics by adding broader, more general medical terms while avoiding overly generic ones.
    """
    blacklist = {"drugs", "medicine", "treatment", "therapy", "biologics", "pharmacology"}
    min_length = 4

    prompt = f"""
    You are a biomedical assistant helping with literature retrieval.
    
    For each of the following biomedical search topics, suggest 1–2 truly *broader* medical terms to improve article search.
    Only include *generalization-level terms* that subsume or encompass the original topic.
    Avoid:
    - Synonyms or related terms
    - Narrower or equivalent terms
    - Overly generic terms like "treatment", "therapy", "biologics", or "medicine"
    
    Example:
    Input: ["asthma", "airway inflammation"]
    Output: ["respiratory diseases", "pulmonary disorders"]
    
    Now expand the following:
    {topics}
    
    Respond only with a valid Python list of strings. No extra text.
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

    expanded_clean = [
        t for t in expanded
        if isinstance(t, str)
        and t.lower() not in blacklist
        and len(t) >= min_length
        and not any(b in t.lower() for b in blacklist)
    ]

    combined = list(set(topics) | set(expanded_clean))

    return combined[:max_terms]


def build_faiss_vectorstore(documents):
    """
    Builds a FAISS vector store from input documents using OpenAI embeddings for efficient similarity search.
    """
    embedding_model = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embedding_model)
    return vectorstore
