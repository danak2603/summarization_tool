import json
import re
from config import OPENAI_API_KEY, client

evaluation_prompt_template = """
You are an expert medical evaluator reviewing the quality of an automatically generated summary.

Please evaluate the summary according to the following criteria. For each criterion, return:
- A score from 1 to 5 (5 = excellent)
- A short explanation of the score (1-2 sentences)

---

User Role: {user_role}

Question: {user_question}

Summary to evaluate:
{summary_text}

Evaluation Criteria:
1. **Relevance to Question** – How well does the summary address the user's question?
2. **Clarity and Structure** – Is the summary clearly written, well-organized, and easy to follow?
3. **Faithfulness to Source** – Does the summary rely on verifiable information from the cited articles? Avoids hallucination?
4. **Citation Accuracy** – Are citations included and do they correspond to the claims made?
5. **User Role Awareness** – Is the summary tailored to the role of the user (e.g., clinician, researcher)?

Please return the evaluation in the following JSON format:
{{
  "relevance_to_question": {{"score": X, "reason": "..."}},
  "clarity_and_structure": {{"score": X, "reason": "..."}},
  "faithfulness_to_source": {{"score": X, "reason": "..."}},
  "citation_accuracy": {{"score": X, "reason": "..."}},
  "role_awareness": {{"score": X, "reason": "..."}}
}}
"""

def clean_json_text(text: str) -> str:
    """
    Cleans a JSON-formatted string by removing Markdown-style code blocks.
    """
    text = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text)
    return text.strip()

def evaluate_summary(user_role: str, user_question: str, summary_text: str):
    """
    Sends the generated summary to an LLM for evaluation based on medical criteria.
    """
    prompt = evaluation_prompt_template.format(
        user_role=user_role,
        user_question=user_question,
        summary_text=summary_text
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a senior biomedical research evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content
    cleaned = clean_json_text(content)

    try:
        evaluation = json.loads(cleaned)
        return evaluation
    except json.JSONDecodeError:
        print("Warning: Couldn't parse the evaluation response. Returning raw text.")
        return content







