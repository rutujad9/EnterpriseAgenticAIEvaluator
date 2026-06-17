import json


GOLDEN_DATASET_PATH = (
    "data/golden_tests.json"
)


def load_golden_dataset():
    try:
        with open(
            GOLDEN_DATASET_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            dataset = json.load(file)

        return dataset

    except FileNotFoundError:
        print(
            "Golden dataset file not found."
        )
        return []

    except json.JSONDecodeError:
        print(
            "Golden dataset JSON is invalid."
        )
        return []