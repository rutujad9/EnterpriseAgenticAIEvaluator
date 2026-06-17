import os

from benchmark_loader import load_golden_dataset
from llm_client import (
    ask_llm_with_metadata,
    get_provider_info,
)
from agents.judge_agent import evaluate_response
from database import save_benchmark_run
from mlflow_logger import log_benchmark_run
from config.prompts import build_worker_prompt, WORKER_PROMPT_VERSION


PROMPT_VERSION = WORKER_PROMPT_VERSION
BENCHMARK_SUITE_VERSION = "golden_50_v1"
JUDGE_PIPELINE_VERSION = "judge_pipeline_v1"
DEEPEVAL_SUITE_VERSION = "deepeval_full_50_v1"


def average_score(scores: list) -> float:
    if not scores:
        return 0.0

    return round(
        sum(scores) / len(scores),
        3,
    )


def run_benchmark():
    provider_info = get_provider_info()
    dataset = load_golden_dataset()

    if not dataset:
        print("No golden tests found.")
        return

    print("\nRunning Golden Dataset Benchmark")
    print("--------------------------------")

    if provider_info["provider"] == "ollama":
        model_name = provider_info["ollama_model"]
    else:
        model_name = provider_info["openai_model"]

    print(f"Provider: {provider_info['provider']}")
    print(f"Model: {model_name}")
    print(f"Prompt Version: {PROMPT_VERSION}")
    print(f"Benchmark Suite: {BENCHMARK_SUITE_VERSION}")
    print("--------------------------------\n")

    total_tests = len(dataset)
    passed_tests = 0
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_estimated_cost_usd = 0.0

    answer_relevancy_scores = []
    hallucination_scores = []

    failed_tests = []

    for test in dataset:
        task = test["task"]
        context = test["context"]
        expected_decision = test["expected_decision"]

        worker_prompt = build_worker_prompt(
            task=task,
            context=context,
        )

        print(f"Running test: {test['id']}")

        worker_result = ask_llm_with_metadata(worker_prompt)
        response = worker_result["response"]

        total_latency += worker_result.get(
            "latency_seconds",
            0,
        )

        total_input_tokens += worker_result.get(
            "input_tokens",
            0,
        )

        total_output_tokens += worker_result.get(
            "output_tokens",
            0,
        )

        total_estimated_cost_usd += worker_result.get(
            "estimated_cost_usd",
            0.0,
        )

        evaluation = evaluate_response(
            task,
            response,
            context,
        )

        answer_relevancy_score = evaluation.get(
            "answer_relevancy_score"
        )

        hallucination_score = evaluation.get(
            "hallucination_score"
        )

        if isinstance(
            answer_relevancy_score,
            (int, float),
        ):
            answer_relevancy_scores.append(
                float(answer_relevancy_score)
            )

        if isinstance(
            hallucination_score,
            (int, float),
        ):
            hallucination_scores.append(
                float(hallucination_score)
            )

        actual_decision = evaluation.get(
            "final_decision",
            "Unknown",
        )

        passed = actual_decision == expected_decision

        if passed:
            passed_tests += 1
            status = "PASS"
        else:
            status = "FAIL"

            failed_tests.append(
                {
                    "id": test["id"],
                    "task": task,
                    "context": context,
                    "expected_decision": expected_decision,
                    "actual_decision": actual_decision,
                    "response": response,
                    "quality_feedback": evaluation.get(
                        "quality_feedback",
                        "",
                    ),
                    "safety_feedback": evaluation.get(
                        "safety_feedback",
                        "",
                    ),
                    "deepeval_feedback": evaluation.get(
                        "deepeval_feedback",
                        "",
                    ),
                    "correctness": evaluation.get("correctness"),
                    "completeness": evaluation.get("completeness"),
                    "clarity": evaluation.get("clarity"),
                    "answer_relevancy_score": answer_relevancy_score,
                    "hallucination_score": hallucination_score,
                }
            )

        print(f"[{status}] {test['id']}")
        print(f"Expected: {expected_decision}")
        print(f"Got: {actual_decision}")
        print(f"Correctness: {evaluation.get('correctness')}")
        print(f"Completeness: {evaluation.get('completeness')}")
        print(f"Clarity: {evaluation.get('clarity')}")
        print("Answer Relevancy:", answer_relevancy_score)
        print("Hallucination Score:", hallucination_score)
        print(
            "Worker Latency:",
            worker_result.get("latency_seconds"),
            "seconds",
        )
        print(
            "Estimated Cost:",
            worker_result.get("estimated_cost_usd", 0.0),
            "USD",
        )
        print("--------------------------------\n")

    accuracy = round(
        (passed_tests / total_tests) * 100,
        2,
    )

    avg_latency = round(
        total_latency / total_tests,
        3,
    )

    total_estimated_cost_usd = round(
        total_estimated_cost_usd,
        6,
    )

    save_benchmark_run(
        provider=provider_info["provider"],
        model=model_name,
        accuracy=accuracy,
        total_tests=total_tests,
        passed_tests=passed_tests,
        avg_latency=avg_latency,
    )

    deepeval_enabled = (
        os.getenv("DEEPEVAL_ENABLED", "false")
        .lower()
        == "true"
    )

    deepeval_cases_evaluated = (
        len(answer_relevancy_scores)
        if deepeval_enabled
        else 0
    )

    deepeval_validation_mode = (
        "full_benchmark_validation"
        if deepeval_enabled
        else "not_run"
    )

    deepeval_answer_relevancy_avg = average_score(
        answer_relevancy_scores
    )

    deepeval_hallucination_avg = average_score(
        hallucination_scores
    )

    log_benchmark_run(
        provider=provider_info["provider"],
        model=model_name,
        passed_tests=passed_tests,
        total_tests=total_tests,
        accuracy=accuracy,
        avg_latency=avg_latency,
        estimated_cost_usd=total_estimated_cost_usd,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        prompt_version=PROMPT_VERSION,
        benchmark_suite_version=BENCHMARK_SUITE_VERSION,
        deepeval_enabled=deepeval_enabled,
        judge_pipeline_version=JUDGE_PIPELINE_VERSION,
        deepeval_suite_version=(
            DEEPEVAL_SUITE_VERSION
            if deepeval_enabled
            else "not_run"
        ),
        deepeval_cases_evaluated=deepeval_cases_evaluated,
        deepeval_validation_mode=deepeval_validation_mode,
        deepeval_answer_relevancy_avg=deepeval_answer_relevancy_avg,
        deepeval_hallucination_avg=deepeval_hallucination_avg,
    )

    print("Benchmark Complete")
    print("------------------")
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Accuracy: {accuracy}%")
    print(f"Average Worker Latency: {avg_latency}s")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Estimated Cost USD: {total_estimated_cost_usd}")
    print(
        "DeepEval Answer Relevancy Avg:",
        deepeval_answer_relevancy_avg,
    )
    print(
        "DeepEval Hallucination Avg:",
        deepeval_hallucination_avg,
    )
    print("Benchmark run saved to database.")
    print("Benchmark run logged to MLflow.")

    if failed_tests:
        print("\nFailed Test Details")
        print("===================")

        for failure in failed_tests:
            print(f"\nTest ID: {failure['id']}")
            print("-------------------")
            print("Task:")
            print(failure["task"])

            print("\nContext:")
            print(failure["context"])

            print("\nExpected Decision:")
            print(failure["expected_decision"])

            print("\nActual Decision:")
            print(failure["actual_decision"])

            print("\nWorker Response:")
            print(failure["response"])

            print("\nScores:")
            print("Correctness:", failure["correctness"])
            print("Completeness:", failure["completeness"])
            print("Clarity:", failure["clarity"])
            print("Answer Relevancy:", failure["answer_relevancy_score"])
            print("Hallucination:", failure["hallucination_score"])

            print("\nQuality Feedback:")
            print(failure["quality_feedback"])

            print("\nSafety Feedback:")
            print(failure["safety_feedback"])

            print("\nDeepEval Feedback:")
            print(failure["deepeval_feedback"])

            print("===================")


if __name__ == "__main__":
    run_benchmark()