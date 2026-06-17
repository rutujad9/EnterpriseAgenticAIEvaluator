import os
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

from config.pricing import estimate_openai_cost


load_dotenv()

provider = os.getenv("LLM_PROVIDER", "openai").lower()
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

LLM_TIMEOUT_SECONDS = int(
    os.getenv(
        "LLM_TIMEOUT_SECONDS",
        "120",
    )
)


def ask_llm(prompt: str) -> str:
    result = ask_llm_with_metadata(prompt)
    return result["response"]


def ask_llm_with_metadata(prompt: str) -> dict:
    start_time = time.perf_counter()
    error = None

    input_tokens = 0
    output_tokens = 0
    estimated_cost_usd = 0.0

    if provider == "openai":
        model = openai_model
        result = ask_openai(prompt)

        response = result["response"]
        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        estimated_cost_usd = result["estimated_cost_usd"]

    elif provider == "ollama":
        model = ollama_model
        response = ask_ollama(prompt)

    else:
        model = "unknown"
        response = (
            "ERROR: Unknown LLM provider. "
            f"Received provider='{provider}'."
        )
        error = response

    if response.startswith("ERROR:"):
        error = response

    latency_seconds = round(
        time.perf_counter() - start_time,
        3,
    )

    return {
        "response": response,
        "provider": provider,
        "model": model,
        "latency_seconds": latency_seconds,
        "error": error,
        "success": error is None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": estimated_cost_usd,
    }


def ask_openai(prompt: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key == "your_api_key_here":
        return {
            "response": (
                "ERROR: Missing OpenAI API key. "
                "Add OPENAI_API_KEY to your .env file."
            ),
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

    try:
        client = OpenAI(
            api_key=api_key,
            timeout=LLM_TIMEOUT_SECONDS,
        )

        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Keep answers clear and practical."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )

        input_tokens = 0
        output_tokens = 0

        if response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0

        estimated_cost_usd = estimate_openai_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=openai_model,
        )

        return {
            "response": (
                response
                .choices[0]
                .message
                .content
                .strip()
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost_usd,
        }

    except Exception as error:
        return {
            "response": (
                "ERROR: OpenAI request failed: "
                f"{error}"
            ),
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
        }


def ask_ollama(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        return response.json().get(
            "response",
            "",
        ).strip()

    except requests.exceptions.Timeout:
        return (
            "ERROR: Ollama request timed out. "
            "The local model took too long to respond."
        )

    except requests.exceptions.RequestException as error:
        return (
            "ERROR: Ollama request failed: "
            f"{error}"
        )


def get_provider_info() -> dict:
    return {
        "provider": provider,
        "openai_model": openai_model,
        "ollama_model": ollama_model,
        "timeout_seconds": LLM_TIMEOUT_SECONDS,
    }