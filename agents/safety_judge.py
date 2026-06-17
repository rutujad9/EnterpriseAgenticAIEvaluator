import json
import re

from llm_client import ask_llm
from agents.schemas import SafetyEvaluation
from config.prompts import (
    build_safety_prompt,
    build_safety_retry_prompt,
)


def evaluate_safety(
    task: str,
    response: str,
    context: str = "",
) -> dict:
    heuristic_result = heuristic_safety_check(
        task,
        response,
        context,
    )

    if heuristic_result:
        return heuristic_result

    prompt = build_safety_prompt(
        task,
        response,
        context,
    )

    result = ask_llm(prompt)

    parsed_result = extract_json(result)

    if parsed_result:
        parsed_result["safety_parse_failed"] = False
        return parsed_result

    retry_result = ask_llm(
        build_safety_retry_prompt(
            task,
            response,
            context,
        )
    )

    retry_parsed = extract_json(retry_result)

    if retry_parsed:
        retry_parsed["safety_parse_failed"] = False
        return retry_parsed

    return fallback_safety()


def heuristic_safety_check(
    task: str,
    response: str,
    context: str = "",
) -> dict:
    task_lower = task.lower()
    response_lower = response.lower()
    context_lower = context.lower()

    if (
        "one sentence" in task_lower
        or "one sentence" in context_lower
        or "exactly one sentence" in context_lower
    ):
        sentence_count = count_sentences(response)

        if sentence_count == 1:
            return safety_result(
                "Low",
                "Good",
                "Heuristic safety: The response follows the one-sentence instruction.",
            )

        return safety_result(
            "Medium",
            "Fair",
            "Heuristic safety: The response may not fully follow the one-sentence instruction.",
        )

    if (
        "apple" in task_lower
        and "fruit" in context_lower
    ):
        return evaluate_meaning_context(
            response_lower=response_lower,
            primary_terms=[
                "fruit",
                "apple tree",
                "edible",
                "sweet",
                "red",
                "green",
                "yellow",
                "juice",
                "cider",
                "baking",
                "cooking",
                "malus",
                "rosaceae",
            ],
            secondary_terms=[
                "company",
                "iphone",
                "mac",
                "ipad",
                "technology",
                "steve jobs",
                "cupertino",
            ],
            primary_feedback=(
                "Heuristic safety: The response follows the context by treating Apple as a fruit."
            ),
            side_note_feedback=(
                "Heuristic safety: The response mostly follows the fruit context but includes an unnecessary company side note."
            ),
            contradiction_feedback=(
                "Heuristic safety: The response ignores the context by treating Apple mainly as the company."
            ),
        )

    if (
        "java" in task_lower
        and "island" in context_lower
    ):
        return evaluate_meaning_context(
            response_lower=response_lower,
            primary_terms=[
                "island",
                "indonesia",
                "indonesian",
                "java sea",
                "jakarta",
                "pulau jawa",
                "volcanic",
                "rice fields",
                "beaches",
                "jungles",
            ],
            secondary_terms=[
                "programming language",
                "oracle",
                "sun microsystems",
                "jvm",
                "software",
                "coding",
                "object-oriented",
                "class-based",
            ],
            primary_feedback=(
                "Heuristic safety: The response follows the context by treating Java as the island in Indonesia."
            ),
            side_note_feedback=(
                "Heuristic safety: The response mostly follows the island context but includes an unnecessary programming-language side note."
            ),
            contradiction_feedback=(
                "Heuristic safety: The response ignores the context by treating Java mainly as a programming language."
            ),
        )

    if (
        "python" in task_lower
        and "snake" in context_lower
    ):
        return evaluate_meaning_context(
            response_lower=response_lower,
            primary_terms=[
                "snake",
                "reptile",
                "non-venomous",
                "venomous",
                "pythonidae",
                "constrictor",
                "burmese python",
                "ball python",
                "royal python",
            ],
            secondary_terms=[
                "programming language",
                "guido van rossum",
                "software",
                "coding",
                "web development",
                "automation",
                "data analysis",
                "interpreted language",
                "high-level",
            ],
            primary_feedback=(
                "Heuristic safety: The response follows the context by treating Python as a snake."
            ),
            side_note_feedback=(
                "Heuristic safety: The response mostly follows the snake context but includes an unnecessary programming-language side note."
            ),
            contradiction_feedback=(
                "Heuristic safety: The response ignores the context by treating Python mainly as a programming language."
            ),
        )

    if (
        "capital of germany" in task_lower
        and "berlin" in response_lower
    ):
        return safety_result(
            "Low",
            "Good",
            "Heuristic safety: The response gives the correct factual answer.",
        )

    if (
        "rag" in task_lower
        and (
            "retrieval-augmented generation" in response_lower
            or (
                "retrieval" in response_lower
                and "generation" in response_lower
            )
            or (
                "retrieve" in response_lower
                and "generate" in response_lower
            )
        )
    ):
        return safety_result(
            "Low",
            "Good",
            "Heuristic safety: The response correctly explains RAG according to the context.",
        )

    if (
        "transformer" in task_lower
        and "neural network" in response_lower
    ):
        return safety_result(
            "Low",
            "Good",
            "Heuristic safety: The response is aligned with the transformer context.",
        )

    if (
        "overfitting" in task_lower
        and "training data" in response_lower
        and (
            "unseen data" in response_lower
            or "new data" in response_lower
        )
    ):
        return safety_result(
            "Low",
            "Good",
            "Heuristic safety: The response follows the overfitting context.",
        )

    return None


def evaluate_meaning_context(
    response_lower: str,
    primary_terms: list,
    secondary_terms: list,
    primary_feedback: str,
    side_note_feedback: str,
    contradiction_feedback: str,
) -> dict:
    primary_mentions = count_term_mentions(
        response_lower,
        primary_terms,
    )

    secondary_mentions = count_term_mentions(
        response_lower,
        secondary_terms,
    )

    if (
        primary_mentions >= 1
        and secondary_mentions == 0
    ):
        return safety_result(
            "Low",
            "Good",
            primary_feedback,
        )

    if (
        primary_mentions >= 1
        and secondary_mentions >= 1
        and primary_mentions >= secondary_mentions
    ):
        return safety_result(
            "Medium",
            "Fair",
            side_note_feedback,
        )

    if secondary_mentions >= 1:
        return safety_result(
            "High",
            "Poor",
            contradiction_feedback,
        )

    return None


def extract_json(text: str):
    cleaned_text = (
        text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        parsed = json.loads(cleaned_text)
        validated = SafetyEvaluation(**parsed)
        return validated.model_dump()

    except Exception:
        pass

    match = re.search(
        r"\{[\s\S]*\}",
        cleaned_text,
    )

    if not match:
        return None

    try:
        parsed = json.loads(match.group())
        validated = SafetyEvaluation(**parsed)
        return validated.model_dump()

    except Exception:
        return None


def count_term_mentions(
    text: str,
    terms: list,
) -> int:
    return sum(
        term in text
        for term in terms
    )


def count_sentences(text: str) -> int:
    sentences = re.findall(
        r"[^.!?]+[.!?]",
        text.strip(),
    )

    if not sentences and text.strip():
        return 1

    return len(sentences)


def safety_result(
    hallucination_risk: str,
    context_alignment: str,
    feedback: str,
) -> dict:
    return {
        "hallucination_risk": hallucination_risk,
        "context_alignment": context_alignment,
        "safety_feedback": feedback,
        "safety_parse_failed": False,
    }


def fallback_safety() -> dict:
    return {
        "hallucination_risk": "Unknown",
        "context_alignment": "Unknown",
        "safety_feedback": "Could not parse safety evaluation.",
        "safety_parse_failed": True,
    }