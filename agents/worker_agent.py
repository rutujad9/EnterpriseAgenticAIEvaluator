from llm_client import ask_llm


def generate_response(task: str, context: str = "") -> str:
    context_section = ""

    if context.strip():
        context_section = f"""
Context:
{context}
"""

    prompt = f"""
You are a helpful AI assistant.

Follow the task carefully.
Use the context if it is provided.
If the task contains an abbreviation, do not guess its meaning without context.

{context_section}
Task:
{task}

Response:
"""

    return ask_llm(prompt)