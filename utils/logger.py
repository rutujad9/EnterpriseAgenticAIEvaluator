import csv
from datetime import datetime
from pathlib import Path


log_file = Path("data/evaluations.csv")

fieldnames = [
    "timestamp",
    "task",
    "context",
    "answer",
    "correctness",
    "completeness",
    "clarity",
    "hallucination_risk",
    "final_decision",
    "short_feedback",
]


def save_evaluation(task: str, context: str, answer: str, evaluation: dict) -> None:
    log_file.parent.mkdir(exist_ok=True)

    write_header = not log_file.exists() or log_file.stat().st_size == 0

    with log_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "task": task,
                "context": context,
                "answer": answer,
                "correctness": evaluation.get("correctness", 0),
                "completeness": evaluation.get("completeness", 0),
                "clarity": evaluation.get("clarity", 0),
                "hallucination_risk": evaluation.get("hallucination_risk", ""),
                "final_decision": evaluation.get("final_decision", ""),
                "short_feedback": evaluation.get("short_feedback", ""),
            }
        )