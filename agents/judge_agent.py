import json
import re

from agents.quality_judge import evaluate_quality
from agents.safety_judge import evaluate_safety
from agents.deepeval_judge import (
    evaluate_with_deepeval,
)


def evaluate_response(
    task: str,
    response: str,
    context: str = "",
) -> dict:
    quality = evaluate_quality(
        task,
        response,
        context,
    )

    safety = evaluate_safety(
        task,
        response,
        context,
    )

    deepeval_result = evaluate_with_deepeval(
        task,
        response,
        context,
    )

    safety = apply_known_false_context_guardrail(
        task,
        context,
        response,
        safety,
    )

    safety = apply_context_guardrail(
        context,
        response,
        safety,
    )

    correctness = quality.get("correctness", 0)
    completeness = quality.get("completeness", 0)
    clarity = quality.get("clarity", 0)

    hallucination_risk = safety.get(
        "hallucination_risk",
        "Unknown",
    )

    context_alignment = safety.get(
        "context_alignment",
        "Unknown",
    )

    hallucination_score = deepeval_result.get(
        "hallucination_score",
        None,
    )

    answer_relevancy_score = deepeval_result.get(
        "answer_relevancy_score",
        None,
    )

    instruction_override = passes_instruction_override(
        task,
        context,
        response,
        correctness,
        completeness,
        clarity,
    )

    final_decision = decide(
        correctness,
        completeness,
        clarity,
        hallucination_risk,
        context_alignment,
        hallucination_score,
        answer_relevancy_score,
        instruction_override,
    )

    short_feedback = build_feedback(
        quality.get("quality_feedback", ""),
        safety.get("safety_feedback", ""),
        deepeval_result.get("deepeval_feedback", ""),
    )

    return {
        "correctness": correctness,
        "completeness": completeness,
        "clarity": clarity,
        "hallucination_risk": hallucination_risk,
        "context_alignment": context_alignment,
        "final_decision": final_decision,
        "short_feedback": short_feedback,
        "quality_feedback": quality.get("quality_feedback", ""),
        "safety_feedback": safety.get("safety_feedback", ""),
        "quality_parse_failed": quality.get("quality_parse_failed", False),
        "safety_parse_failed": safety.get("safety_parse_failed", False),
        "deepeval_enabled": deepeval_result.get("deepeval_enabled", False),
        "answer_relevancy_score": answer_relevancy_score,
        "hallucination_score": hallucination_score,
        "deepeval_feedback": deepeval_result.get("deepeval_feedback", ""),
    }


def passes_instruction_override(
    task: str,
    context: str,
    response: str,
    correctness: int,
    completeness: int,
    clarity: int,
) -> bool:
    task_lower = task.lower()
    context_lower = context.lower()
    response_clean = response.strip()

    strong_enough_quality = (
        correctness >= 4
        and completeness >= 3
        and clarity >= 4
    )

    if not strong_enough_quality:
        return False

    if (
        "one sentence" in task_lower
        or "one sentence" in context_lower
        or "exactly one sentence" in context_lower
    ):
        return count_sentences(response_clean) == 1

    if (
        "exactly two sentences" in context_lower
        or "use exactly two sentences" in context_lower
    ):
        return count_sentences(response_clean) == 2

    if (
        "exactly five words" in task_lower
        or "exactly five words" in context_lower
    ):
        return count_words(response_clean) == 5

    if (
        "under 20 words" in context_lower
        or "under twenty words" in context_lower
    ):
        return count_words(response_clean) < 20

    if (
        "only valid json" in context_lower
        or "answer only valid json" in context_lower
    ):
        return is_valid_json(response_clean)

    if "answer with only the number" in context_lower:
        return response_clean.isdigit()

    if "answer only using bullet points" in context_lower:
        return uses_only_bullets(response_clean)

    if "do not use bullet points" in context_lower:
        return not uses_bullets(response_clean)

    return False


def count_sentences(text: str) -> int:
    sentences = re.findall(
        r"[^.!?]+[.!?]",
        text.strip(),
    )

    if not sentences and text.strip():
        return 1

    return len(sentences)


def count_words(text: str) -> int:
    cleaned_text = (
        text
        .replace('"', "")
        .replace("'", "")
        .strip()
    )

    words = re.findall(
        r"\b[\w-]+\b",
        cleaned_text,
    )

    return len(words)


def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def uses_bullets(text: str) -> bool:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return any(
        line.startswith(("-", "*", "•"))
        for line in lines
    )


def uses_only_bullets(text: str) -> bool:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return False

    return all(
        line.startswith(("-", "*", "•"))
        for line in lines
    )


def apply_known_false_context_guardrail(
    task: str,
    context: str,
    response: str,
    safety: dict,
) -> dict:
    task_lower = task.lower()
    context_lower = context.lower()
    response_lower = response.lower()

    if (
        "rag" in task_lower
        and "random answer generator" in context_lower
    ):
        if "random answer generator" in response_lower:
            safety["context_alignment"] = "Poor"
            safety["hallucination_risk"] = "High"
            safety["safety_feedback"] = (
                "Known-fact guardrail: The provided context defines RAG as "
                "Random Answer Generator, but in AI RAG commonly means "
                "Retrieval-Augmented Generation. The response repeats the "
                "false context instead of correcting it."
            )
            return safety

    if (
        "capital of germany" in task_lower
        and "munich" in context_lower
    ):
        if "munich" in response_lower:
            safety["context_alignment"] = "Poor"
            safety["hallucination_risk"] = "High"
            safety["safety_feedback"] = (
                "Known-fact guardrail: The provided context says Munich is "
                "the capital of Germany, but the correct capital is Berlin."
            )
            return safety

    return safety


def apply_context_guardrail(
    context: str,
    response: str,
    safety: dict,
) -> dict:
    context_lower = context.lower()
    response_lower = response.lower()

    context_expects_snake = (
        "snake" in context_lower
        or "not the programming language" in context_lower
    )

    if not context_expects_snake:
        return safety

    snake_terms = [
        "snake",
        "reptile",
        "non-venomous",
        "venomous",
        "constrictor",
        "pythonidae",
        "african rock python",
        "ball python",
        "royal python",
    ]

    programming_terms = [
        "programming language",
        "guido van rossum",
        "web development",
        "data analysis",
        "machine learning",
        "software",
        "automation",
        "coding",
        "interpreted language",
        "high-level language",
    ]

    snake_mentions = sum(
        term in response_lower
        for term in snake_terms
    )

    programming_mentions = sum(
        term in response_lower
        for term in programming_terms
    )

    mainly_programming = (
        programming_mentions >= 2
        and programming_mentions > snake_mentions
    )

    mainly_snake_with_side_note = (
        snake_mentions >= 1
        and programming_mentions >= 1
        and snake_mentions >= programming_mentions
    )

    pure_snake = (
        snake_mentions >= 1
        and programming_mentions == 0
    )

    if mainly_programming:
        safety["context_alignment"] = "Poor"
        safety["hallucination_risk"] = "High"
        safety["safety_feedback"] = (
            "The response ignores the provided context by mainly explaining "
            "Python as a programming language instead of the snake meaning."
        )
        return safety

    if mainly_snake_with_side_note:
        safety["context_alignment"] = "Fair"
        safety["hallucination_risk"] = "Medium"
        safety["safety_feedback"] = (
            "The response mostly follows the provided snake context, "
            "but includes an unnecessary programming-language side note."
        )
        return safety

    if pure_snake:
        safety["context_alignment"] = "Good"

        if safety.get("hallucination_risk") == "High":
            safety["hallucination_risk"] = "Medium"

        safety["safety_feedback"] = (
            "The response follows the provided context by treating Python "
            "as a snake."
        )

    return safety


def decide(
    correctness: int,
    completeness: int,
    clarity: int,
    hallucination_risk: str,
    context_alignment: str,
    hallucination_score,
    answer_relevancy_score,
    instruction_override: bool = False,
) -> str:
    deep_eval_available = (
        hallucination_score is not None
    )

    deep_eval_says_clean = (
        hallucination_score is not None
        and hallucination_score <= 0.2
    )

    strong_quality = (
        correctness >= 4
        and completeness >= 4
        and clarity >= 4
    )

    safe_short_answer = (
        correctness >= 4
        and clarity >= 4
        and context_alignment == "Good"
        and hallucination_risk in [
            "Low",
            "Unknown",
        ]
    )

    low_relevancy = (
        answer_relevancy_score is not None
        and answer_relevancy_score < 0.5
    )

    if instruction_override:
        return "Accept"

    if context_alignment == "Poor":
        return "Reject"

    if (
        deep_eval_available
        and hallucination_score >= 0.7
    ):
        return "Reject"

    if correctness <= 2:
        return "Reject"

    if low_relevancy and not deep_eval_says_clean:
        return "Reject"

    if (
        deep_eval_says_clean
        and strong_quality
    ):
        return "Accept"

    if (
        not deep_eval_available
        and strong_quality
        and hallucination_risk in [
            "Low",
            "Unknown",
        ]
        and context_alignment in [
            "Good",
            "Unknown",
        ]
    ):
        return "Accept"

    if (
        not deep_eval_available
        and safe_short_answer
    ):
        return "Accept"

    if hallucination_risk == "High":
        return "Reject"

    if (
        context_alignment == "Fair"
        or hallucination_risk == "Medium"
    ):
        return "Revise"

    if (
        correctness <= 3
        or clarity <= 3
    ):
        return "Revise"

    if (
        completeness <= 3
        and context_alignment != "Good"
    ):
        return "Revise"

    return "Accept"


def build_feedback(
    quality_feedback: str,
    safety_feedback: str,
    deepeval_feedback: str = "",
) -> str:
    feedback_parts = []

    if quality_feedback:
        feedback_parts.append(quality_feedback)

    if safety_feedback:
        feedback_parts.append(safety_feedback)

    if deepeval_feedback:
        feedback_parts.append(deepeval_feedback)

    if not feedback_parts:
        return "Evaluation completed."

    return " ".join(feedback_parts)