"""
Unit Tests — Bedrock KB Client & Utilities
Run with: pytest tests/ -v
"""

import pytest
import json
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from utils import sanitize_query, format_citations


# ── Sanitize Query Tests ───────────────────────────────────────────────────────
class TestSanitizeQuery:
    def test_strips_whitespace(self):
        assert sanitize_query("  hello world  ") == "hello world"

    def test_collapses_internal_whitespace(self):
        assert sanitize_query("what  is   the policy") == "what is the policy"

    def test_removes_null_bytes(self):
        assert "\x00" not in sanitize_query("hello\x00world")

    def test_enforces_max_length(self):
        long_query = "a" * 2000
        result = sanitize_query(long_query, max_length=100)
        assert len(result) == 100

    def test_empty_string(self):
        assert sanitize_query("") == ""

    def test_normal_query_unchanged(self):
        q = "What is the remote work policy?"
        assert sanitize_query(q) == q


# ── Format Citations Tests ─────────────────────────────────────────────────────
class TestFormatCitations:
    def test_empty_citations_returns_empty_string(self):
        assert format_citations([]) == ""

    def test_single_citation_rendered(self):
        citations = [{"source": "policy.pdf", "text": "Employees may work remotely.", "score": 0.9}]
        html = format_citations(citations)
        assert "policy.pdf" in html
        assert "Sources" in html

    def test_deduplicates_same_source(self):
        citations = [
            {"source": "doc.pdf", "text": "Section 1.", "score": 0.9},
            {"source": "doc.pdf", "text": "Section 2.", "score": 0.8},
        ]
        html = format_citations(citations)
        assert html.count("doc.pdf") == 1

    def test_score_hidden_by_default(self):
        citations = [{"source": "doc.pdf", "text": "Test.", "score": 0.75}]
        html = format_citations(citations, show_scores=False)
        assert "75%" not in html

    def test_score_shown_when_enabled(self):
        citations = [{"source": "doc.pdf", "text": "Test.", "score": 0.75}]
        html = format_citations(citations, show_scores=True)
        assert "75%" in html

    def test_xss_escaped(self):
        citations = [{"source": "<script>alert(1)</script>", "text": "Test.", "score": 0.5}]
        html = format_citations(citations)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ── Bedrock Client Tests (mocked) ──────────────────────────────────────────────
class TestBedrockKBClient:

    MOCK_RESPONSE = {
        "output": {"text": "The remote work policy allows up to 3 days per week."},
        "citations": [
            {
                "retrievedReferences": [
                    {
                        "content": {"text": "Employees may work remotely up to 3 days per week."},
                        "location": {"s3Location": {"uri": "s3://bucket/documents/hr-policy.pdf"}},
                        "score": 0.92,
                    }
                ]
            }
        ],
        "sessionId": "test-session-123",
    }

    @patch("bedrock_client.boto3.Session")
    def test_query_returns_structured_response(self, mock_session):
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client
        mock_client.retrieve_and_generate.return_value = self.MOCK_RESPONSE

        from bedrock_client import BedrockKBClient
        client = BedrockKBClient(knowledge_base_id="test-kb-123", region_name="us-east-1")
        result = client.query("What is the remote work policy?")

        assert "answer" in result
        assert "citations" in result
        assert result["chunks_retrieved"] == 1
        assert "3 days per week" in result["answer"]

    @patch("bedrock_client.boto3.Session")
    def test_parse_response_extracts_filename(self, mock_session):
        mock_session.return_value.client.return_value = MagicMock()

        from bedrock_client import BedrockKBClient
        client = BedrockKBClient("test-kb", "us-east-1")
        parsed = client._parse_response(self.MOCK_RESPONSE)

        assert parsed["citations"][0]["source"] == "hr-policy.pdf"

    @patch("bedrock_client.boto3.Session")
    def test_empty_citations(self, mock_session):
        mock_session.return_value.client.return_value = MagicMock()
        response = {"output": {"text": "No relevant docs found."}, "citations": [], "sessionId": ""}

        from bedrock_client import BedrockKBClient
        client = BedrockKBClient("test-kb", "us-east-1")
        parsed = client._parse_response(response)

        assert parsed["chunks_retrieved"] == 0
        assert parsed["citations"] == []
