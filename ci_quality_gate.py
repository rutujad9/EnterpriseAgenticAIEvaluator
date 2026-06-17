import json
from pathlib import Path


GOLDEN_TESTS_PATH = Path("data/golden_tests.json")
BENCHMARK_RESULTS_PATH = Path("reports/benchmark_results.json")


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_golden_tests():
    tests = load_json(
        GOLDEN_TESTS_PATH
    )

    if not isinstance(
        tests,
        list,
    ):
        raise ValueError(
            "Golden tests file "
            "must contain a list."
        )

    if len(tests) < 50:
        raise ValueError(
            "Expected at least "
            f"50 golden tests, "
            f"found {len(tests)}."
        )

    required_fields = {
        "id",
        "task",
        "context",
        "expected_decision",
        "category",
    }

    categories = set()

    for test in tests:

        missing_fields = (
            required_fields
            - set(test.keys())
        )

        if missing_fields:
            raise ValueError(
                f"Test "
                f"{test.get('id', 'UNKNOWN')} "
                f"missing fields: "
                f"{missing_fields}"
            )

        if (
            test[
                "expected_decision"
            ]
            not in {
                "Accept",
                "Reject",
                "Revise",
            }
        ):
            raise ValueError(
                f"Invalid "
                "expected_decision "
                f"in test "
                f"{test['id']}: "
                f"{test['expected_decision']}"
            )

        categories.add(
            test["category"]
        )

    if len(categories) < 5:
        raise ValueError(
            "Golden benchmark "
            "should cover at least "
            "5 categories."
        )

    print(
        "Golden dataset "
        f"check passed: "
        f"{len(tests)} tests."
    )

    print(
        "Categories covered: "
        f"{sorted(categories)}"
    )


def validate_benchmark_results():

    results = load_json(
        BENCHMARK_RESULTS_PATH
    )

    total_tests = results.get(
        "total_tests",
        0,
    )

    if total_tests < 50:
        raise ValueError(
            "Benchmark must "
            "contain at least "
            f"50 tests, "
            f"found "
            f"{total_tests}."
        )

    models = results.get(
        "models",
        [],
    )

    if not models:
        raise ValueError(
            "No model benchmark "
            "results found."
        )

    required_models = {
        "mistral",
        "phi3",
        "llama3.2",
    }

    tested_models = {
        model_result["model"]
        for model_result
        in models
    }

    missing_models = (
        required_models
        - tested_models
    )

    if missing_models:
        raise ValueError(
            "Missing benchmark "
            "results for models: "
            f"{missing_models}"
        )

    best_model = max(
        models,
        key=lambda model_result:
        model_result["accuracy"],
    )

    minimum_required_accuracy = (
        results.get(
            "minimum_required_accuracy",
            70.0,
        )
    )

    if (
        best_model["accuracy"]
        < minimum_required_accuracy
    ):
        raise ValueError(
            "Quality gate failed: "
            "best model accuracy "
            f"{best_model['accuracy']}% "
            "is below required "
            f"{minimum_required_accuracy}%."
        )

    if (
        best_model["model"]
        != results.get(
            "best_local_model"
        )
    ):
        raise ValueError(
            "Best local model "
            "in report does not "
            "match benchmark "
            "results."
        )

    print(
        "Benchmark quality "
        "gate passed: "
        f"{best_model['model']} "
        f"achieved "
        f"{best_model['accuracy']}%."
    )


def main():
    validate_golden_tests()
    validate_benchmark_results()

    print(
        "CI quality gate "
        "passed successfully."
    )


if __name__ == "__main__":
    main()
