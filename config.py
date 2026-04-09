"""
Application Configuration
Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    # ── Required: Set these in your environment or .env file ──────────────────
    KNOWLEDGE_BASE_ID: str = os.getenv("BEDROCK_KB_ID", "YOUR_KB_ID_HERE")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")

    # ── Model Configuration ────────────────────────────────────────────────────
    # Claude 3 Sonnet is the recommended balance of quality vs cost
    # Alternatives:
    #   anthropic.claude-3-haiku-20240307-v1:0  (faster, cheaper)
    #   anthropic.claude-3-opus-20240229-v1:0   (highest quality, most expensive)
    MODEL_ID: str = os.getenv(
        "BEDROCK_MODEL_ID",
        "anthropic.claude-3-sonnet-20240229-v1:0"
    )

    @property
    def MODEL_ARN(self) -> str:
        return f"arn:aws:bedrock:{self.AWS_REGION}::foundation-model/{self.MODEL_ID}"

    # ── Retrieval Defaults ─────────────────────────────────────────────────────
    DEFAULT_NUM_RESULTS: int = int(os.getenv("DEFAULT_NUM_RESULTS", "5"))
    DEFAULT_MIN_SCORE: float = float(os.getenv("DEFAULT_MIN_SCORE", "0.3"))

    # ── App Settings ──────────────────────────────────────────────────────────
    APP_TITLE: str = "Enterprise Knowledge Base"
    MAX_QUERY_LENGTH: int = 1000
    QUERY_HISTORY_LIMIT: int = 50
