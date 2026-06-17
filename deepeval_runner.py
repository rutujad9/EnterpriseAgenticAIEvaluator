import json
from statistics import mean

from benchmark_loader import load_golden_dataset
from llm_client import ask_llm_with_metadata
from agents.judge_agent import evaluate_response
from config.prompts import build_worker_prompt


DEEPEVAL_SAMPLE_SIZE = 10


def run_deepeval_sample():
    dataset = load_golden_dataset()

    selected_tests = dataset[:DEEPEVAL_SAMPLE_SIZE]

    answer_relevancy_scores = []
    hallucination_scores = []

    print("\nRunning Controlled DeepEval Sample")
    print("----------------------------------")

    for test in selected_tests:
        print(f"Running: {test['id']}")

        worker_prompt = build_worker_prompt(
            task=test["task"],
            context=test["context"],
        )

        worker_result = ask_llm_with_metadata(
            worker_prompt
        )

        response = worker_result["response"]

        evaluation = evaluate_response(
            task=test["task"],
            response=response,
            context=test["context"],
        )

        relevancy = evaluation.get(
            "answer_relevancy_score"
        )

        hallucination = evaluation.get(
            "hallucination_score"
        )

        if relevancy is not None:
            answer_relevancy_scores.append(
                relevancy
            )

        if hallucination is not None:
            hallucination_scores.append(
                hallucination
            )

    summary = {
        "deepeval_enabled": True,
        "deepeval_suite_version":
            "deepeval_high_risk_v1",
        "mode": "controlled_sample",
        "cases_evaluated":
            DEEPEVAL_SAMPLE_SIZE,
        "planned_cases":
            DEEPEVAL_SAMPLE_SIZE,
        "metrics": {
            "faithfulness_avg":
                "Phase 2",
            "answer_relevancy_avg":
                round(
                    mean(
                        answer_relevancy_scores
                    ),
                    3,
                )
                if answer_relevancy_scores
                else None,
            "hallucination_avg":
                round(
                    mean(
                        hallucination_scores
                    ),
                    3,
                )
                if hallucination_scores
                else None,
            "context_alignment_avg":
                "Phase 2",
        },
        "status": "completed",
        "note":
            (
                "Controlled 10-case "
                "DeepEval validation "
                "completed using "
                "gpt-4o-mini."
            ),
    }

    with open(
        "reports/deepeval_summary.json",
        "w",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\nDeepEval summary updated."
    )


if __name__ == "__main__":
    run_deepeval_sample()
