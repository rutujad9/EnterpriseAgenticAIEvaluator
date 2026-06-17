import os

from deepeval.metrics import (
    AnswerRelevancyMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase


def evaluate_with_deepeval(
    task: str,
    response: str,
    context: str = "",
) -> dict:
    deepeval_enabled = (
        os.getenv(
            "DEEPEVAL_ENABLED",
            "false",
        )
        .lower()
        == "true"
    )

    if not deepeval_enabled:
        return {
            "deepeval_enabled": False,
            "answer_relevancy_score": None,
            "hallucination_score": None,
            "deepeval_feedback": (
                "DeepEval disabled by DEEPEVAL_ENABLED=false."
            ),
        }

    try:
        retrieval_context = (
            [context]
            if context.strip()
            else []
        )

        test_case = LLMTestCase(
            input=task,
            actual_output=response,
            retrieval_context=retrieval_context,
            context=retrieval_context,
        )

        answer_relevancy_metric = (
            AnswerRelevancyMetric(
                threshold=0.7,
            )
        )

        hallucination_metric = (
            HallucinationMetric(
                threshold=0.5,
            )
        )

        answer_relevancy_metric.measure(
            test_case
        )

        hallucination_metric.measure(
            test_case
        )

        deepeval_feedback = (
            "Answer Relevancy: "
            f"{answer_relevancy_metric.reason} "
            "Hallucination: "
            f"{hallucination_metric.reason}"
        )

        return {
            "deepeval_enabled": True,
            "answer_relevancy_score": (
                answer_relevancy_metric.score
            ),
            "hallucination_score": (
                hallucination_metric.score
            ),
            "deepeval_feedback": (
                deepeval_feedback
            ),
        }

    except Exception as error:
        return {
            "deepeval_enabled": False,
            "answer_relevancy_score": None,
            "hallucination_score": None,
            "deepeval_feedback": (
                f"DeepEval error: {error}"
            ),
        }