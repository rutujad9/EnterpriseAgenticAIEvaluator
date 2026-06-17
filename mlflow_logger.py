from pathlib import Path
import mlflow


MLFLOW_DIR = Path("mlruns").resolve()

mlflow.set_tracking_uri(
    f"file://{MLFLOW_DIR}"
)

mlflow.set_experiment(
    "Enterprise_AI_Evaluator"
)


def log_benchmark_run(
    provider: str,
    model: str,
    passed_tests: int,
    total_tests: int,
    accuracy: float,
    avg_latency: float = 0.0,
    estimated_cost_usd: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    prompt_version: str = "v1",
    benchmark_suite_version: str = "golden_50_v1",
    deepeval_enabled: bool = False,
    judge_pipeline_version: str = "judge_pipeline_v1",
    deepeval_suite_version: str = "not_run",
    deepeval_cases_evaluated: int = 0,
    deepeval_validation_mode: str = "not_run",
    deepeval_answer_relevancy_avg: float = 0.0,
    deepeval_hallucination_avg: float = 0.0,
):
    with mlflow.start_run():

        mlflow.set_tag(
            "mlflow.note.content",
            (
                f"50-test adversarial benchmark "
                f"for model={model}, "
                f"provider={provider}, "
                f"prompt={prompt_version}, "
                f"suite={benchmark_suite_version}, "
                f"deepeval={deepeval_enabled}. "
                f"Tracks accuracy, latency, "
                f"cost, reliability, and "
                f"DeepEval validation metadata."
            ),
        )

        mlflow.log_param("provider", provider)
        mlflow.log_param("model", model)
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("benchmark_suite_version", benchmark_suite_version)
        mlflow.log_param("deepeval_enabled", deepeval_enabled)
        mlflow.log_param("judge_pipeline_version", judge_pipeline_version)
        mlflow.log_param("deepeval_suite_version", deepeval_suite_version)
        mlflow.log_param("deepeval_validation_mode", deepeval_validation_mode)

        mlflow.log_metric("passed_tests", passed_tests)
        mlflow.log_metric("failed_tests", total_tests - passed_tests)
        mlflow.log_metric("total_tests", total_tests)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("avg_latency_seconds", avg_latency)
        mlflow.log_metric("estimated_cost_usd", estimated_cost_usd)
        mlflow.log_metric("input_tokens", input_tokens)
        mlflow.log_metric("output_tokens", output_tokens)
        mlflow.log_metric(
            "deepeval_cases_evaluated",
            deepeval_cases_evaluated,
        )
        mlflow.log_metric(
            "deepeval_answer_relevancy_avg",
            deepeval_answer_relevancy_avg,
        )
        mlflow.log_metric(
            "deepeval_hallucination_avg",
            deepeval_hallucination_avg,
        )

        mlflow.set_tag("project", "EnterpriseAgenticAIEvaluator")
        mlflow.set_tag("system_type", "AI Evaluation and Observability Platform")
        mlflow.set_tag("benchmark_type", "adversarial golden test suite")
        mlflow.set_tag("tracking_purpose", "model reliability comparison")

        print(
            f"MLflow logged: {model} "
            f"({accuracy}%) "
            f"cost=${estimated_cost_usd} "
            f"prompt={prompt_version} "
            f"suite={benchmark_suite_version} "
            f"deepeval={deepeval_enabled}"
        )