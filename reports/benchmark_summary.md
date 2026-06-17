# Enterprise Agentic AI Evaluator — Benchmark Summary

## Objective

This project evaluates AI-generated responses using a multi-agent judging workflow focused on:

- correctness
- completeness
- clarity
- hallucination risk
- context alignment

The goal is to benchmark local and cloud LLMs for reliability, latency, and evaluation quality using adversarial golden tests.

---

## Golden Benchmark Suite

Benchmark suite:

50-test adversarial golden dataset

Categories covered:

- AI concepts
- ML concepts
- context ambiguity
- contradiction handling
- prompt injection
- hallucination traps
- instruction following
- multi-constraint reasoning
- safety (financial, legal, medical)

Purpose:

To measure how reliably models follow instructions, avoid hallucinations, handle context, and make safe decisions.

---

## Local Model Benchmark Results

| Model | Provider | Accuracy | Passed Tests | Failed Tests |
|--------|----------|----------|--------------|---------------|
| mistral | ollama | 74–78% | 37–39 / 50 | 11–13 |
| phi3 | ollama | 50% | 25 / 50 | 25 |
| llama3.2 | ollama | 44% | 22 / 50 | 28 |

### Key Finding

Mistral consistently achieved the best reliability among local models and was selected as the default local inference model.

Tradeoff:

- Higher quality than phi3 and llama3.2
- Stronger context handling
- Better prompt-following behavior
- Slightly higher latency

---

## Reliability Engineering

The platform includes:

- Pydantic structured outputs
- retry-on-malformed JSON
- heuristic fallback evaluation
- CI/CD quality gates
- benchmark regression protection

Current CI threshold:

Minimum benchmark accuracy:

70%

Purpose:

Prevent silent quality regressions during development.

---

## Observability

Tracked metrics include:

- worker latency
- judge latency
- total latency
- P50 latency
- P95 latency
- accept/reject/revise rates
- parse failures
- benchmark accuracy trends

Purpose:

Production-style monitoring for AI system reliability.

---

## MLflow Experiment Tracking

MLflow is used to track:

- provider
- model
- benchmark accuracy
- latency
- prompt version
- benchmark suite version
- judge pipeline version
- DeepEval configuration

Example tracked metadata:

- prompt_version = v1
- benchmark_suite_version = golden_50_v1
- judge_pipeline_version = judge_pipeline_v1

Purpose:

Enable reproducible benchmarking and experiment comparison.

---

## DeepEval Integration

DeepEval integration is supported through configuration flags.

Planned metrics:

- Faithfulness
- Answer Relevancy
- Hallucination
- Context Alignment

To reduce unnecessary API cost, DeepEval evaluation is designed to run on selected high-risk benchmark cases instead of every evaluation.

---

## Current Production Features

Implemented:

- Multi-agent evaluation workflow
- Ollama + OpenAI provider support
- Golden benchmark suite
- SQLite persistence
- Prompt versioning
- MLflow experiment tracking
- CI/CD quality gates
- Reliability monitoring
- P50/P95 latency observability
- Streamlit analytics dashboard

