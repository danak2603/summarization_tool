from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import OPENAI_API_KEY

def get_system_prompt_by_user_role(user_role: str) -> str:
    """
    Returns a role-specific system prompt that guides the LLM to tailor responses based on the user's clinical or scientific background.
    """
    role = user_role.lower()

    if role in {"general practitioner", "gp", "primary care physician", "family doctor", "doctor"}:
        return (
            "You are a biomedical assistant helping a general physician. "
            "Provide clinically relevant, concise answers suitable for a non-specialist physician. "
            "Focus on actionable guidance and clarity over depth."
        )
    elif role in {"pediatrician", "pediatric physician"}:
        return (
            "You are a biomedical assistant helping a pediatrician. "
            "Summarize treatments with a focus on child-specific considerations, "
            "including safety, efficacy, and dosing."
        )
    elif role in {"rheumatologist", "oncologist", "cardiologist", "neurologist"}:
        return (
            f"You are a biomedical assistant helping a specialist {user_role}. "
            "Provide detailed, evidence-based summaries with technical medical language and "
            "focus on the latest therapeutic advances and clinical relevance."
        )
    elif role in {"biomedical researcher", "researcher", "scientist"}:
        return (
            "You are a biomedical assistant helping a biomedical researcher. "
            "Focus on mechanisms, study design, and emerging treatment targets. "
            "Use technical terminology where appropriate."
        )
    else:
        return (
            f"You are a biomedical assistant helping a {user_role}. "
            "Adjust the language and focus based on their likely level of clinical or scientific expertise. "
            "Make sure to cite articles and stay clear and concise."
        )

def generate_chat_prompt(user_role, user_question, retrieved_docs):
    """
    Constructs a list of chat messages (system + user) combining the research question and relevant document content for summarization.
    """
    context_snippets = ""
    for i, doc in enumerate(retrieved_docs, 1):
        pmid = doc.metadata.get("pmid")
        pmcid = doc.metadata.get("pmcid")

        if pmid and not pmid.startswith("PMID"):
            source = f"PMID{pmid}"
        elif pmcid and not pmcid.startswith("PMC"):
            source = f"PMC{pmcid}"
        else:
            source = pmid or pmcid or f"Doc{i}"

        title = doc.metadata.get("title", "Unknown Title")
        context_snippets += f"[{source}] {title}\n{doc.page_content.strip()}\n\n"

    system_message = SystemMessage(
        content=get_system_prompt_by_user_role(user_role)
    )
    human_message = HumanMessage(
        content=(
            f"Research Question:\n{user_question}\n\n"
            f"Scientific Articles:\n{context_snippets}\n\n"
            "Please write a concise and informative summary (≤5,000 characters), "
            f"tailored for a {user_role}, and cite sources using [PMID...] or [PMC...].\n\n"
            "Focus **only** on answering the research question. "
            "Prioritize treatment options and clinically actionable insights. "
            "Avoid including unrelated background information, pathophysiology, etiology, or diagnosis "
            "unless directly relevant to understanding treatment decisions. "
            "The summary should be useful for a clinician in practice."
        )
    )

    return [system_message, human_message]


def generate_summary_from_documents(user_role, user_question, retrieved_docs):
    """
    Sends the formatted prompt and retrieved documents to the LLM to generate a concise, role-specific summary.
    """
    chat_messages = generate_chat_prompt(user_role, user_question, retrieved_docs)
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    response_chat = llm.invoke(chat_messages)
    return response_chat.content




