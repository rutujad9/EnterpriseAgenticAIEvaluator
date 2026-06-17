import sqlite3
from datetime import datetime

import pandas as pd


DATABASE_PATH = "data/evaluations.db"


def create_database():
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            task TEXT,
            context TEXT,
            response TEXT,
            correctness INTEGER,
            completeness INTEGER,
            clarity INTEGER,
            hallucination_risk TEXT,
            context_alignment TEXT,
            final_decision TEXT,
            short_feedback TEXT,
            quality_feedback TEXT,
            safety_feedback TEXT,
            provider TEXT,
            model TEXT,
            worker_latency REAL,
            judge_latency REAL,
            total_latency REAL,
            quality_parse_failed INTEGER,
            safety_parse_failed INTEGER,
            deepeval_enabled INTEGER,
            answer_relevancy_score REAL,
            hallucination_score REAL,
            deepeval_feedback TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            provider TEXT,
            model TEXT,
            accuracy REAL,
            total_tests INTEGER,
            passed_tests INTEGER,
            avg_latency REAL
        )
        """
    )

    add_missing_columns(cursor)

    connection.commit()
    connection.close()


def add_missing_columns(cursor):
    cursor.execute("PRAGMA table_info(evaluations)")
    existing_columns = [column[1] for column in cursor.fetchall()]

    new_columns = {
        "provider": "TEXT",
        "model": "TEXT",
        "worker_latency": "REAL",
        "judge_latency": "REAL",
        "total_latency": "REAL",
        "quality_parse_failed": "INTEGER",
        "safety_parse_failed": "INTEGER",
        "deepeval_enabled": "INTEGER",
        "answer_relevancy_score": "REAL",
        "hallucination_score": "REAL",
        "deepeval_feedback": "TEXT",
    }

    for column_name, column_type in new_columns.items():
        if column_name not in existing_columns:
            cursor.execute(
                f"""
                ALTER TABLE evaluations
                ADD COLUMN {column_name}
                {column_type}
                """
            )


def save_evaluation_to_db(
    task: str,
    context: str,
    response: str,
    evaluation: dict,
    run_metadata: dict = None,
):
    create_database()

    if run_metadata is None:
        run_metadata = {}

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO evaluations (
            timestamp,
            task,
            context,
            response,
            correctness,
            completeness,
            clarity,
            hallucination_risk,
            context_alignment,
            final_decision,
            short_feedback,
            quality_feedback,
            safety_feedback,
            provider,
            model,
            worker_latency,
            judge_latency,
            total_latency,
            quality_parse_failed,
            safety_parse_failed,
            deepeval_enabled,
            answer_relevancy_score,
            hallucination_score,
            deepeval_feedback
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?
        )
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            task,
            context,
            response,
            evaluation.get("correctness", 0),
            evaluation.get("completeness", 0),
            evaluation.get("clarity", 0),
            evaluation.get("hallucination_risk", ""),
            evaluation.get("context_alignment", ""),
            evaluation.get("final_decision", ""),
            evaluation.get("short_feedback", ""),
            evaluation.get("quality_feedback", ""),
            evaluation.get("safety_feedback", ""),
            run_metadata.get("provider", ""),
            run_metadata.get("model", ""),
            run_metadata.get("worker_latency", 0),
            run_metadata.get("judge_latency", 0),
            run_metadata.get("total_latency", 0),
            int(evaluation.get("quality_parse_failed", False)),
            int(evaluation.get("safety_parse_failed", False)),
            int(evaluation.get("deepeval_enabled", False)),
            evaluation.get("answer_relevancy_score"),
            evaluation.get("hallucination_score"),
            evaluation.get("deepeval_feedback", ""),
        ),
    )

    connection.commit()
    connection.close()


def save_benchmark_run(
    provider: str,
    model: str,
    accuracy: float,
    total_tests: int,
    passed_tests: int,
    avg_latency: float,
):
    create_database()

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO benchmark_runs (
            timestamp,
            provider,
            model,
            accuracy,
            total_tests,
            passed_tests,
            avg_latency
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            provider,
            model,
            accuracy,
            total_tests,
            passed_tests,
            avg_latency,
        ),
    )

    connection.commit()
    connection.close()


def load_benchmark_runs():
    create_database()

    connection = sqlite3.connect(DATABASE_PATH)

    benchmark_runs = pd.read_sql_query(
        """
        SELECT *
        FROM benchmark_runs
        ORDER BY id DESC
        """,
        connection,
    )

    connection.close()

    return benchmark_runs


def load_evaluations():
    create_database()

    connection = sqlite3.connect(DATABASE_PATH)

    evaluations = pd.read_sql_query(
        """
        SELECT *
        FROM evaluations
        ORDER BY id DESC
        """,
        connection,
    )

    connection.close()

    return evaluations


def clear_evaluations():
    create_database()

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("DELETE FROM evaluations")

    connection.commit()
    connection.close()