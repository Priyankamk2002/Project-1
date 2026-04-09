"""
Amazon Bedrock Knowledge Base Client
Handles all interactions with Bedrock Knowledge Bases and the Retrieve & Generate API
"""

import boto3
import json
import logging
from typing import Optional
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class BedrockKBClient:
    """
    Client for Amazon Bedrock Knowledge Bases.
    
    Uses the RetrieveAndGenerate API which:
    1. Converts the query to an embedding vector
    2. Performs semantic search against the vector store
    3. Passes retrieved chunks as context to the LLM
    4. Returns a grounded, cited answer
    """

    def __init__(
        self,
        knowledge_base_id: str,
        region_name: str = "us-east-1",
        model_arn: Optional[str] = None,
    ):
        self.knowledge_base_id = knowledge_base_id
        self.region_name = region_name
        self.model_arn = model_arn or (
            f"arn:aws:bedrock:{region_name}::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
        )

        try:
            # Uses IAM role when running on EC2; falls back to ~/.aws/credentials locally
            session = boto3.Session(region_name=region_name)
            self.bedrock_agent_runtime = session.client(
                service_name="bedrock-agent-runtime",
                region_name=region_name,
            )
            logger.info(f"Bedrock client initialized for KB: {knowledge_base_id}")
        except NoCredentialsError:
            raise RuntimeError(
                "AWS credentials not found. On EC2, ensure the instance has an IAM role. "
                "Locally, run 'aws configure' or set environment variables."
            )

    # ── Core RAG Query ─────────────────────────────────────────────────────────
    def query(
        self,
        question: str,
        num_results: int = 5,
        min_score: float = 0.0,
        search_type: str = "HYBRID",
    ) -> dict:
        """
        Submit a question to the Knowledge Base using RetrieveAndGenerate.
        
        Args:
            question: Natural language question from the user
            num_results: Max number of source chunks to retrieve
            min_score: Minimum relevance score filter (0.0–1.0)
            search_type: HYBRID | SEMANTIC | KEYWORD
            
        Returns:
            {
                "answer": str,
                "citations": [{"text": str, "source": str, "score": float}],
                "chunks_retrieved": int,
                "session_id": str,
            }
        """
        try:
            retrieval_config = {
                "vectorSearchConfiguration": {
                    "numberOfResults": num_results,
                    "overrideSearchType": search_type,
                }
            }

            # Apply score filter if threshold > 0
            if min_score > 0:
                retrieval_config["vectorSearchConfiguration"]["filter"] = {
                    "greaterThan": {
                        "key": "_score",
                        "value": min_score,
                    }
                }

            response = self.bedrock_agent_runtime.retrieve_and_generate(
                input={"text": question},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": self.knowledge_base_id,
                        "modelArn": self.model_arn,
                        "retrievalConfiguration": retrieval_config,
                        "generationConfiguration": {
                            "promptTemplate": {
                                "textPromptTemplate": self._get_system_prompt()
                            },
                            "inferenceConfig": {
                                "textInferenceConfig": {
                                    "maxTokens": 1024,
                                    "temperature": 0.1,   # Low temp for factual accuracy
                                    "topP": 0.9,
                                }
                            },
                        },
                    },
                },
            )

            return self._parse_response(response)

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ResourceNotFoundException":
                raise RuntimeError(
                    f"Knowledge Base '{self.knowledge_base_id}' not found. "
                    "Verify the KB ID and region in your configuration."
                )
            elif error_code == "AccessDeniedException":
                raise RuntimeError(
                    "Access denied. Ensure your IAM role has 'bedrock:RetrieveAndGenerate' permission."
                )
            elif error_code == "ValidationException":
                raise RuntimeError(f"Invalid request: {e.response['Error']['Message']}")
            else:
                raise RuntimeError(f"Bedrock API error ({error_code}): {e.response['Error']['Message']}")

    # ── Retrieve-Only (no generation) ──────────────────────────────────────────
    def retrieve_only(
        self,
        query: str,
        num_results: int = 5,
        search_type: str = "HYBRID",
    ) -> list[dict]:
        """
        Retrieve relevant document chunks without LLM generation.
        Useful for debugging retrieval quality.
        
        Returns list of: {"text": str, "source": str, "score": float}
        """
        response = self.bedrock_agent_runtime.retrieve(
            knowledgeBaseId=self.knowledge_base_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": num_results,
                    "overrideSearchType": search_type,
                }
            },
        )

        chunks = []
        for result in response.get("retrievalResults", []):
            chunks.append({
                "text": result["content"]["text"],
                "source": result.get("location", {}).get("s3Location", {}).get("uri", "Unknown"),
                "score": result.get("score", 0.0),
            })

        return chunks

    # ── Response Parser ────────────────────────────────────────────────────────
    def _parse_response(self, raw_response: dict) -> dict:
        """Parse the raw Bedrock RetrieveAndGenerate response."""
        output = raw_response.get("output", {})
        answer = output.get("text", "No answer generated.")

        citations = []
        for citation in raw_response.get("citations", []):
            for ref in citation.get("retrievedReferences", []):
                source_uri = (
                    ref.get("location", {})
                       .get("s3Location", {})
                       .get("uri", "Unknown source")
                )
                # Convert S3 URI to readable filename
                source_name = source_uri.split("/")[-1] if "/" in source_uri else source_uri

                citations.append({
                    "text": ref.get("content", {}).get("text", ""),
                    "source": source_name,
                    "source_uri": source_uri,
                    "score": ref.get("score", 0.0),
                })

        return {
            "answer": answer,
            "citations": citations,
            "chunks_retrieved": len(citations),
            "session_id": raw_response.get("sessionId", ""),
        }

    # ── System Prompt ──────────────────────────────────────────────────────────
    def _get_system_prompt(self) -> str:
        return """You are an expert enterprise knowledge assistant. Your role is to answer employee questions accurately using ONLY the retrieved company documents provided as context.

Guidelines:
- Base your answer strictly on the provided context. Never fabricate information.
- If the context doesn't contain enough information to answer confidently, say so clearly.
- Be concise but complete. Use bullet points for multi-step information.
- When referencing specific policies or figures, be precise.
- If multiple documents provide conflicting information, acknowledge the discrepancy.
- Always maintain a professional, helpful tone.

$search_results$

Question: $query$

Answer:"""
