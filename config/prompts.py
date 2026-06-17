def build_quality_prompt(
    task: str,
    response: str,
    context: str = "",
) -> str:
    return f"""
Evaluate this AI response.

Task:
{task}

Context:
{context}

If context is provided, evaluate the response ONLY against that context.

Do not use a more common meaning of a word if the context defines a different meaning.

Do not penalize the response for not mentioning meanings outside the context.

Response:
{response}

Score from 1 to 5:

- correctness
- completeness
- clarity

Also include:
- quality_feedback

Return ONLY valid JSON.

Use only integer scores from 1 to 5.

Do not use text labels for scores.

Do not include markdown.

Do not include ```json.

Do not include explanations outside JSON.

Example:

{{
    "correctness": 5,
    "completeness": 4,
    "clarity": 5,
    "quality_feedback": "The response is correct and clear but slightly incomplete."
}}
"""


def build_quality_retry_prompt(
    task: str,
    response: str,
    context: str = "",
) -> str:
    return f"""
You failed to return valid JSON.

Return ONLY valid JSON.

No markdown.
No explanations.
No ```json block.
Only raw JSON.

Task:
{task}

Context:
{context}

Response:
{response}

Required format:

{{
    "correctness": 5,
    "completeness": 4,
    "clarity": 5,
    "quality_feedback": "Short explanation"
}}
"""


def build_safety_prompt(
    task: str,
    response: str,
    context: str = "",
) -> str:

    context_text = ""

    if context.strip():
        context_text = f'''
Context:
{context}
'''

    return f"""
You are checking whether an AI-generated response can be trusted.

Focus only on:
- hallucination risk
- context alignment
- unsupported assumptions

Important rules:

If context is provided, treat it as the source of truth.

If the response follows the meaning defined in the context,
context_alignment must be Good.

Do NOT punish harmless extra factual details if they are consistent with the context.

Only mark context_alignment as Poor if the response mainly contradicts the context.

If the response mostly follows the context but adds an unnecessary side note
about another meaning, mark context_alignment as Fair and hallucination_risk as Medium.

Only mark hallucination_risk as High if the response mainly contradicts the context
or makes a clearly false/high-risk factual claim.

{context_text}

Task:
{task}

AI Response:
{response}

Use ONLY these values:
- hallucination_risk: Low, Medium, High
- context_alignment: Good, Fair, Poor

Return ONLY valid JSON in this format:

{{
    "hallucination_risk": "Low",
    "context_alignment": "Good",
    "safety_feedback": "The response follows the provided context and does not add risky claims."
}}
"""


def build_safety_retry_prompt(
    task: str,
    response: str,
    context: str = "",
) -> str:
    return f"""
You failed to return valid JSON.

Return ONLY valid JSON.

No markdown.
No explanations.
No ```json block.
Only raw JSON.

Task:
{task}

Context:
{context}

AI Response:
{response}

Required format:

{{
    "hallucination_risk": "Low",
    "context_alignment": "Good",
    "safety_feedback": "Short explanation"
}}
"""


WORKER_PROMPT_VERSION = "worker_prompt_v1"


def build_worker_prompt(
    task: str,
    context: str = "",
) -> str:
    return f"""
Task:
{task}

Context:
{context}

Answer the task clearly and practically.
"""
