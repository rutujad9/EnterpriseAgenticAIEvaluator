from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


SCHEMA_VERSION = "v1"


class QualityEvaluation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    schema_version: str = SCHEMA_VERSION

    correctness: int = Field(
        ge=1,
        le=5,
    )

    completeness: int = Field(
        ge=1,
        le=5,
    )

    clarity: int = Field(
        ge=1,
        le=5,
    )

    quality_feedback: str = Field(
        min_length=3,
        max_length=1000,
    )


class SafetyEvaluation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    schema_version: str = SCHEMA_VERSION

    hallucination_risk: Literal[
        "Low",
        "Medium",
        "High",
    ]

    context_alignment: Literal[
        "Good",
        "Fair",
        "Poor",
    ]

    safety_feedback: str = Field(
        min_length=3,
        max_length=1000,
    )


class FinalEvaluation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    provider: str
    model: str

    correctness: int = Field(
        ge=1,
        le=5,
    )

    completeness: int = Field(
        ge=1,
        le=5,
    )

    clarity: int = Field(
        ge=1,
        le=5,
    )

    hallucination_risk: Literal[
        "Low",
        "Medium",
        "High",
    ]

    context_alignment: Literal[
        "Good",
        "Fair",
        "Poor",
    ]

    final_decision: Literal[
        "Accept",
        "Revise",
        "Reject",
    ]

    feedback: str = Field(
        min_length=3,
        max_length=2000,
    )

    worker_latency: float = Field(
        ge=0,
    )

    judge_latency: float = Field(
        ge=0,
    )

    total_latency: float = Field(
        ge=0,
    )