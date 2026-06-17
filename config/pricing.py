OPENAI_PRICING = {
    "gpt-4o-mini": {
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
    },
    "gpt-4o": {
        "input_per_1m": 5.00,
        "output_per_1m": 15.00,
    },
}


def estimate_openai_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> float:
    pricing = OPENAI_PRICING.get(model)

    if not pricing:
        return 0.0

    input_cost = (
        input_tokens / 1_000_000
    ) * pricing["input_per_1m"]

    output_cost = (
        output_tokens / 1_000_000
    ) * pricing["output_per_1m"]

    total_cost = (
        input_cost +
        output_cost
    )

    return round(total_cost, 6)
