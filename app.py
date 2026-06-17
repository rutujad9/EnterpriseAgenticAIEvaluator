import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from agents.judge_agent import evaluate_response
from database import (
    clear_evaluations,
    load_benchmark_runs,
    load_evaluations,
    save_evaluation_to_db,
)
from llm_client import ask_llm_with_metadata, get_provider_info
from config.prompts import build_worker_prompt


st.set_page_config(
    page_title="Enterprise Agentic AI Evaluator",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Enterprise Agentic AI Evaluator")

st.markdown(
    """
    Production-style platform for evaluating AI-generated responses using
    multi-agent judging, golden benchmarks, DeepEval validation,
    MLflow experiment tracking, and reliability observability.
    """
)

provider_info = get_provider_info()

if provider_info["provider"] == "ollama":
    st.caption(f"Running with Ollama model: {provider_info['ollama_model']}")
else:
    st.caption(f"Running with OpenAI model: {provider_info['openai_model']}")

st.divider()

with st.sidebar:
    st.header("Project Overview")

    st.markdown(
        """
        ## Enterprise Agentic AI Evaluator

        Production-style platform for **AI response evaluation,
        reliability engineering, benchmarking, and observability**.

        ### Multi-Agent Workflow
        - **Worker Agent** → generates responses
        - **Judge Agent** → evaluates quality & safety
        - **Evaluation Pipeline** → stores, benchmarks, and tracks outcomes

        ### Evaluation Dimensions
        **Correctness • Completeness • Clarity • Hallucination Risk • Context Alignment**

        ### Enterprise Features
        - Ollama + OpenAI provider support
        - Multi-agent structured evaluation
        - Pydantic structured outputs + retry logic
        - Golden benchmark suite (50 adversarial tests)
        - DeepEval validation framework
        - MLflow experiment tracking
        - Prompt versioning & CI/CD quality gate
        - P50/P95 latency monitoring
        - Token & cost observability

        ---
        Built to demonstrate **production AI evaluation systems,
        LLM reliability engineering, benchmarking, and observability**.
        """
    )

task = st.text_area(
    "📝 Evaluation Task",
    placeholder="Example: Explain Retrieval-Augmented Generation (RAG) in simple terms.",
    height=100,
)

context = st.text_area(
    "📚 Optional Context / Expected Meaning",
    placeholder="Example: RAG refers to Retrieval-Augmented Generation in AI.",
    height=100,
)

if st.button("🚀 Run Evaluation"):
    if not task.strip():
        st.warning("Please enter a task first.")
    else:
        worker_prompt = build_worker_prompt(
            task=task,
            context=context,
        )

        start_time = time.perf_counter()

        with st.spinner("Worker agent is generating a response..."):
            worker_result = ask_llm_with_metadata(worker_prompt)
            answer = worker_result["response"]

        worker_latency = worker_result["latency_seconds"]

        st.subheader("🤖 Worker Agent Response")
        st.write(answer)

        with st.spinner("Judge agent is evaluating the response..."):
            judge_start = time.perf_counter()
            evaluation = evaluate_response(task, answer, context)
            judge_latency = round(time.perf_counter() - judge_start, 3)

        total_latency = round(time.perf_counter() - start_time, 3)

        run_metadata = {
            "provider": worker_result["provider"],
            "model": worker_result["model"],
            "worker_latency": worker_latency,
            "judge_latency": judge_latency,
            "total_latency": total_latency,
        }

        save_evaluation_to_db(
            task,
            context,
            answer,
            evaluation,
            run_metadata,
        )

        st.subheader("🧪 AI Evaluation Result")

        col1, col2, col3 = st.columns(3)

        col1.metric("Correctness", evaluation.get("correctness", 0))
        col2.metric("Completeness", evaluation.get("completeness", 0))
        col3.metric("Clarity", evaluation.get("clarity", 0))

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Hallucination Risk",
            evaluation.get("hallucination_risk", "Unknown"),
        )

        col5.metric(
            "Context Alignment",
            evaluation.get("context_alignment", "Unknown"),
        )

        col6.metric(
            "Final Decision",
            evaluation.get("final_decision", "Unknown"),
        )

        st.write("**Provider:**", worker_result["provider"])
        st.write("**Model:**", worker_result["model"])
        st.write("**Worker Latency:**", f"{worker_latency} seconds")
        st.write("**Judge Latency:**", f"{judge_latency} seconds")
        st.write("**Total Latency:**", f"{total_latency} seconds")

        st.write(
            "**Feedback:**",
            evaluation.get("short_feedback", "No feedback available."),
        )

        with st.expander("🔍 Evaluation Details"):
            st.write(
                "**Quality Judge:**",
                evaluation.get("quality_feedback", "No quality feedback."),
            )

            st.write(
                "**Safety Judge:**",
                evaluation.get("safety_feedback", "No safety feedback."),
            )

            st.divider()

            st.write("### 🧱 Structured Output Reliability")

            st.write(
                "**Quality Parse Failed:**",
                evaluation.get("quality_parse_failed", False),
            )

            st.write(
                "**Safety Parse Failed:**",
                evaluation.get("safety_parse_failed", False),
            )

            st.divider()

            st.write("### 🧪 DeepEval Metrics")

            st.write(
                "**Enabled:**",
                evaluation.get("deepeval_enabled", False),
            )

            st.write(
                "**Answer Relevancy Score:**",
                evaluation.get("answer_relevancy_score", "Not available"),
            )

            st.write(
                "**Hallucination Score:**",
                evaluation.get("hallucination_score", "Not available"),
            )

            st.write(
                "**DeepEval Feedback:**",
                evaluation.get("deepeval_feedback", "No DeepEval feedback."),
            )

st.divider()
st.subheader("📈 Evaluation Analytics")

if st.button("Clear Evaluation History"):
    clear_evaluations()
    st.success("Evaluation history cleared.")
    st.rerun()

try:
    evaluations = load_evaluations()
    benchmark_runs = load_benchmark_runs()

    if evaluations.empty:
        st.info("No evaluations saved yet.")
    else:
        avg_correctness = round(evaluations["correctness"].mean(), 2)
        avg_completeness = round(evaluations["completeness"].mean(), 2)
        avg_clarity = round(evaluations["clarity"].mean(), 2)

        m1, m2, m3 = st.columns(3)

        m1.metric("Avg Correctness", avg_correctness)
        m2.metric("Avg Completeness", avg_completeness)
        m3.metric("Avg Clarity", avg_clarity)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🛡 Reliability & Safety Overview")

        total_evaluations = len(evaluations)
        decision_counts = evaluations["final_decision"].value_counts()

        accept_rate = round(
            decision_counts.get("Accept", 0) / total_evaluations * 100,
            1,
        )

        reject_rate = round(
            decision_counts.get("Reject", 0) / total_evaluations * 100,
            1,
        )

        revise_rate = round(
            decision_counts.get("Revise", 0) / total_evaluations * 100,
            1,
        )

        quality_parse_failures = (
            int(evaluations["quality_parse_failed"].sum())
            if "quality_parse_failed" in evaluations.columns
            else 0
        )

        safety_parse_failures = (
            int(evaluations["safety_parse_failed"].sum())
            if "safety_parse_failed" in evaluations.columns
            else 0
        )

        deepeval_enabled_count = (
            int(evaluations["deepeval_enabled"].sum())
            if "deepeval_enabled" in evaluations.columns
            else 0
        )

        reliability_table = pd.DataFrame(
            [
                {
                    "Metric": "Accept Rate",
                    "Value": f"{accept_rate}%",
                },
                {
                    "Metric": "Reject Rate",
                    "Value": f"{reject_rate}%",
                },
                {
                    "Metric": "Revise Rate",
                    "Value": f"{revise_rate}%",
                },
                {
                    "Metric": "Structured Output Failures",
                    "Value": quality_parse_failures + safety_parse_failures,
                },
                {
                    "Metric": "Quality JSON Failures",
                    "Value": quality_parse_failures,
                },
                {
                    "Metric": "Safety JSON Failures",
                    "Value": safety_parse_failures,
                },
                {
                    "Metric": "DeepEval Runs",
                    "Value": deepeval_enabled_count,
                },
            ]
        )

        reliability_table["Value"] = reliability_table["Value"].astype(str)

        st.dataframe(
            reliability_table,
            width="stretch",
            hide_index=True,
        )

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("📊 Score Trends")

        if len(evaluations) >= 2:
            recent_evaluations = evaluations.tail(20)

            chart_data = recent_evaluations[
                [
                    "correctness",
                    "completeness",
                    "clarity",
                ]
            ].reset_index(drop=True)

            st.line_chart(chart_data, height=300)
            st.caption("Showing last 20 evaluations.")
        else:
            st.info("Run at least 2 evaluations to see score trends.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("⚖️ Decision Distribution")

        decision_chart_data = pd.DataFrame(
            {
                "Decision": decision_counts.index.astype(str),
                "Count": decision_counts.values,
            }
        )

        decision_chart = (
            alt.Chart(decision_chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Decision:N",
                    title="Decision",
                    sort=["Accept", "Revise", "Reject"],
                    axis=alt.Axis(
                        labelAngle=0,
                    ),
                ),
                y=alt.Y(
                    "Count:Q",
                    title="Count",
                ),
                tooltip=["Decision", "Count"],
            )
            .properties(
                height=320,
            )
        )

        st.altair_chart(
            decision_chart,
            use_container_width=True,
        )

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("⏱ Latency Overview")

        avg_worker_latency = round(evaluations["worker_latency"].mean(), 2)
        avg_judge_latency = round(evaluations["judge_latency"].mean(), 2)
        avg_total_latency = round(evaluations["total_latency"].mean(), 2)

        p50_worker_latency = round(evaluations["worker_latency"].quantile(0.50), 2)
        p95_worker_latency = round(evaluations["worker_latency"].quantile(0.95), 2)

        p50_judge_latency = round(evaluations["judge_latency"].quantile(0.50), 2)
        p95_judge_latency = round(evaluations["judge_latency"].quantile(0.95), 2)

        p50_total_latency = round(evaluations["total_latency"].quantile(0.50), 2)
        p95_total_latency = round(evaluations["total_latency"].quantile(0.95), 2)

        latency_table = pd.DataFrame(
            [
                {
                    "Metric": "Average Worker Latency",
                    "Value": f"{avg_worker_latency}s",
                },
                {
                    "Metric": "Average Judge Latency",
                    "Value": f"{avg_judge_latency}s",
                },
                {
                    "Metric": "Average Total Latency",
                    "Value": f"{avg_total_latency}s",
                },
                {
                    "Metric": "Worker P50 / P95",
                    "Value": f"{p50_worker_latency}s / {p95_worker_latency}s",
                },
                {
                    "Metric": "Judge P50 / P95",
                    "Value": f"{p50_judge_latency}s / {p95_judge_latency}s",
                },
                {
                    "Metric": "Total P50 / P95",
                    "Value": f"{p50_total_latency}s / {p95_total_latency}s",
                },
            ]
        )

        latency_table["Value"] = latency_table["Value"].astype(str)

        st.dataframe(
            latency_table,
            width="stretch",
            hide_index=True,
        )

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("🏆 Model Reliability Benchmark")

        if benchmark_runs.empty:
            st.info("No benchmark runs saved yet. Run python benchmark_runner.py first.")
        else:
            full_benchmark_runs = benchmark_runs[
                benchmark_runs["total_tests"] == 50
            ].copy()

            if full_benchmark_runs.empty:
                st.warning(
                    "No full 50-test benchmark runs found yet. "
                    "Run benchmark_runner.py with the 50-test golden suite."
                )
            else:
                latest_full_runs = (
                    full_benchmark_runs
                    .sort_values(
                         by=[
                                "provider",
                                "model",
                                "accuracy",
                                "avg_latency",
                                "id",
                            ],
                            ascending=[
                                            True,
                                            True,
                                            False,
                                            True,
                                            False,
                                        ],
                    )
                    .drop_duplicates(subset=["provider", "model"])
                    .copy()
                )

                latest_full_runs["failed_tests"] = (
                    latest_full_runs["total_tests"]
                    - latest_full_runs["passed_tests"]
                )

                latest_full_runs = latest_full_runs.sort_values(
                    by=[
                        "accuracy",
                        "avg_latency",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                ).reset_index(drop=True)

                medals = [
                    "🥇",
                    "🥈",
                    "🥉",
                ]

                latest_full_runs["rank"] = [
                    medals[index]
                    if index < len(medals)
                    else f"#{index + 1}"
                    for index in range(len(latest_full_runs))
                ]

                benchmark_leaderboard = latest_full_runs[
                    [
                        "rank",
                        "provider",
                        "model",
                        "accuracy",
                        "passed_tests",
                        "failed_tests",
                        "total_tests",
                        "avg_latency",
                        "timestamp",
                    ]
                ].rename(
                    columns={
                        "rank": "Rank",
                        "provider": "Provider",
                        "model": "Model",
                        "accuracy": "Accuracy (%)",
                        "passed_tests": "Correct Evaluations",
                        "failed_tests": "Failed Evaluations",
                        "total_tests": "Total Tests",
                        "avg_latency": "Avg Latency (s)",
                        "timestamp": "Last Run",
                    }
                )

                benchmark_leaderboard[
                    [
                        "Accuracy (%)",
                        "Avg Latency (s)",
                    ]
                ] = benchmark_leaderboard[
                    [
                        "Accuracy (%)",
                        "Avg Latency (s)",
                    ]
                ].round(2)

                st.dataframe(
                    benchmark_leaderboard,
                    width="stretch",
                    hide_index=True,
                )

                best_model = benchmark_leaderboard.iloc[0]

                st.success(
                    f"Best benchmark result: {best_model['Model']} "
                    f"with {best_model['Accuracy (%)']}% accuracy "
                    f"({best_model['Correct Evaluations']}/"
                    f"{best_model['Total Tests']} correct evaluations)."
                )

                st.caption(
                    "Only full 50-test benchmark runs are shown here. "
                    "Older 3-test and 10-test experiments are intentionally hidden."
                )

                st.markdown("<br><br>", unsafe_allow_html=True)
                st.subheader("📊 Benchmark Comparison Charts")

                c1, c2 = st.columns(2)

                with c1:
                    st.write("### Accuracy by Model")

                    accuracy_chart_data = (
                        latest_full_runs[
                            [
                                "model",
                                "accuracy",
                            ]
                        ]
                        .copy()
                        .sort_values(
                            "accuracy",
                            ascending=False,
                        )
                    )

                    accuracy_chart_data["accuracy"] = (
                        accuracy_chart_data["accuracy"].round(2)
                    )

                    accuracy_chart = (
                        alt.Chart(accuracy_chart_data)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "accuracy:Q",
                                title="Accuracy (%)",
                            ),
                            y=alt.Y(
                                "model:N",
                                title="Model",
                                sort="-x",
                            ),
                            tooltip=[
                                "model",
                                "accuracy",
                            ],
                        )
                    )

                    accuracy_chart = accuracy_chart.properties(
                        height=420,
                    )

                    st.altair_chart(
                        accuracy_chart,
                        use_container_width=True,
                    )

                with c2:
                    st.write("### Latency by Model")

                    latency_chart_data = (
                        latest_full_runs[
                            [
                                "model",
                                "avg_latency",
                            ]
                        ]
                        .copy()
                        .sort_values(
                            "avg_latency",
                        )
                    )

                    latency_chart_data["avg_latency"] = (
                        latency_chart_data["avg_latency"].round(2)
                    )

                    latency_chart = (
                        alt.Chart(latency_chart_data)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "avg_latency:Q",
                                title="Average latency (s)",
                            ),
                            y=alt.Y(
                                "model:N",
                                title="Model",
                                sort="x",
                            ),
                            tooltip=[
                                "model",
                                "avg_latency",
                            ],
                        )
                    )

                    latency_chart = latency_chart.properties(
                        height=420,
                    )

                    st.altair_chart(
                        latency_chart,
                        use_container_width=True,
                    )

        st.subheader("🌐 Provider Benchmark")

        if benchmark_runs.empty:
            st.info("No benchmark runs saved yet.")
        else:
            full_benchmark_runs = benchmark_runs[
                benchmark_runs["total_tests"] == 50
            ].copy()

            if full_benchmark_runs.empty:
                st.info(
                    "No full 50-test provider benchmark data yet. "
                    "OpenAI will appear here after a complete 50-test benchmark run."
                )
            else:
                latest_model_runs = (
                    full_benchmark_runs
                    .sort_values("id", ascending=False)
                    .drop_duplicates(subset=["provider", "model"])
                )

                provider_benchmark = (
                    latest_model_runs
                    .groupby("provider")
                    .agg(
                        avg_accuracy=("accuracy", "mean"),
                        best_accuracy=("accuracy", "max"),
                        avg_latency=("avg_latency", "mean"),
                        models_tested=("model", "count"),
                    )
                    .reset_index()
                )

                provider_benchmark[
                    [
                        "avg_accuracy",
                        "best_accuracy",
                        "avg_latency",
                    ]
                ] = provider_benchmark[
                    [
                        "avg_accuracy",
                        "best_accuracy",
                        "avg_latency",
                    ]
                ].round(2)

                st.dataframe(
                    provider_benchmark,
                    width="stretch",
                    hide_index=True,
                )


        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("🧪 DeepEval Reliability Validation")

        deepeval_report_path = Path(
            "reports/deepeval_summary.json"
        )

        if deepeval_report_path.exists():

            import json

            with open(
                deepeval_report_path,
                "r",
            ) as file:
                deepeval_report = json.load(file)

            metrics = deepeval_report.get(
                "metrics",
                {},
            )

            deepeval_summary_table = pd.DataFrame(
                [
                    {
                        "Metric": "DeepEval enabled",
                        "Value": str(
                            deepeval_report.get(
                                "deepeval_enabled",
                                False,
                            )
                        ),
                    },
                    {
                        "Metric": "Suite version",
                        "Value": deepeval_report.get(
                            "deepeval_suite_version",
                            "Unknown",
                        ),
                    },
                    {
                        "Metric": "Validation mode",
                        "Value": deepeval_report.get(
                            "mode",
                            "Unknown",
                        ),
                    },
                    {
                        "Metric": "Cases evaluated",
                        "Value": (
                            f"{deepeval_report.get('cases_evaluated', 0)}/"
                            f"{deepeval_report.get('planned_cases', 0)}"
                        ),
                    },
                    {
                        "Metric": "Answer relevancy avg",
                        "Value": str(metrics.get(
                            "answer_relevancy_avg",
                            "Not available",
                        )),
                    },
                    {
                        "Metric": "Hallucination avg",
                        "Value": str(metrics.get(
                            "hallucination_avg",
                            "Not available",
                        )),
                    },
                    {
                        "Metric": "Context alignment",
                        "Value": str(metrics.get(
                            "context_alignment_avg",
                            "Covered by Judge",
                        )),
                    },
                ]
            )

            deepeval_summary_table["Value"] = deepeval_summary_table["Value"].astype(str)

            st.dataframe(
                deepeval_summary_table,
                width="stretch",
                hide_index=True,
            )

            st.caption(
                deepeval_report.get(
                    "note",
                    "",
                )
            )

        else:
            st.info(
                "DeepEval report not found."
            )

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("📊 MLflow Experiment Tracking & Reproducibility")

        mlflow_dir = Path("mlruns")
        mlflow_enabled = mlflow_dir.exists()

        mlflow_summary_table = pd.DataFrame(
            [
                {
                    "Item": "Tracking tool",
                    "Value": "MLflow",
                },
                {
                    "Item": "Tracking status",
                    "Value": "Enabled"
                    if mlflow_enabled
                    else "No local runs yet",
                },
                {
                    "Item": "Experiment",
                    "Value": "Enterprise_AI_Evaluator",
                },
                {
                    "Item": "Local tracking store",
                    "Value": "./mlruns",
                },
            ]
        )

        mlflow_summary_table["Value"] = mlflow_summary_table["Value"].astype(str)

        st.dataframe(
            mlflow_summary_table,
            width="stretch",
            hide_index=True,
        )

        if not benchmark_runs.empty:
            full_benchmark_runs = benchmark_runs[
                benchmark_runs["total_tests"] == 50
            ].copy()

            if not full_benchmark_runs.empty:
                latest_logged_run = (
                    full_benchmark_runs
                    .sort_values("id", ascending=False)
                    .iloc[0]
                )

                st.success(
                    "Latest MLflow-tracked benchmark: "
                    f"{latest_logged_run['model']} "
                    f"with {latest_logged_run['accuracy']}% accuracy "
                    f"({latest_logged_run['passed_tests']}/"
                    f"{latest_logged_run['total_tests']} correct evaluations)."
                )

        st.write("Open the MLflow UI:")

        st.code(
            "mlflow ui --backend-store-uri file://$(pwd)/mlruns",
            language="bash",
        )

        st.caption(
            "MLflow tracks benchmark experiments, model versions, prompt versions, "
            "latency, cost, and DeepEval validation metrics."
        )

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("📈 Benchmark Accuracy Over Time")

        if benchmark_runs.empty:
            st.info("No benchmark history available.")
        else:
            full_benchmark_runs = benchmark_runs[
                benchmark_runs["total_tests"] == 50
            ].copy()

            if full_benchmark_runs.empty:
                st.info("No full 50-test benchmark history available.")
            else:
                history_data = (
                    full_benchmark_runs
                    .sort_values("id")
                    .tail(30)
                    .copy()
                )

                history_data["Run"] = [
                    f"Run {index + 1}"
                    for index in range(len(history_data))
                ]

                history_chart = (
                    alt.Chart(history_data)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "Run:N",
                            title="Benchmark run",
                            axis=alt.Axis(
                                labelAngle=0,
                            ),
                        ),
                        y=alt.Y(
                            "accuracy:Q",
                            title="Accuracy (%)",
                        ),
                        color=alt.Color(
                            "model:N",
                            title="Model",
                        ),
                        tooltip=[
                            "Run",
                            "model",
                            "provider",
                            "accuracy",
                            "timestamp",
                        ],
                    )
                    .properties(
                        height=420,
                    )
                )

                st.altair_chart(
                    history_chart,
                    use_container_width=True,
                )

                st.caption("Showing latest full 50-test benchmark runs only.")

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("📜 Evaluation History & Audit Trail")

        display_columns = [
            "timestamp",
            "provider",
            "model",
            "task",
            "final_decision",
            "correctness",
            "completeness",
            "clarity",
            "hallucination_risk",
            "context_alignment",
            "total_latency",
            "deepeval_enabled",
            "answer_relevancy_score",
            "hallucination_score",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in evaluations.columns
        ]

        st.caption("Latest 20 saved evaluations with model, decision, latency, reliability, and DeepEval metadata.")

        st.dataframe(
            evaluations[available_columns].head(20),
            width="stretch",
        )

except (FileNotFoundError, pd.errors.EmptyDataError):
    st.info("No saved data yet.")