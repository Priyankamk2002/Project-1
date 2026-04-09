#!/usr/bin/env python3
"""
Amazon Bedrock Knowledge Base Setup Script
==========================================
Creates a Bedrock Knowledge Base backed by:
  - Amazon OpenSearch Serverless (vector store)
  - Amazon S3 (document source)
  - Amazon Titan Embeddings v2 (embedding model)

Prerequisites:
  pip install boto3
  AWS credentials with AdministratorAccess or the specific KB permissions

Usage:
  python setup_knowledge_base.py --s3-bucket YOUR_BUCKET --region us-east-1

After completion, add the printed KB_ID to your .env file as BEDROCK_KB_ID.
"""

import boto3
import json
import time
import argparse
import sys
from botocore.exceptions import ClientError

# ── Argument Parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Create Bedrock Knowledge Base")
parser.add_argument("--s3-bucket",   required=True, help="S3 bucket containing documents")
parser.add_argument("--s3-prefix",   default="documents/", help="S3 prefix (folder)")
parser.add_argument("--kb-name",     default="enterprise-knowledge-base")
parser.add_argument("--region",      default="us-east-1")
parser.add_argument("--embedding-model", 
                    default="amazon.titan-embed-text-v2:0",
                    help="Embedding model to use")
args = parser.parse_args()

# ── Clients ────────────────────────────────────────────────────────────────────
session      = boto3.Session(region_name=args.region)
bedrock_agent = session.client("bedrock-agent")
iam           = session.client("iam")
oss           = session.client("opensearchserverless")
sts           = session.client("sts")

ACCOUNT_ID  = sts.get_caller_identity()["Account"]
REGION      = args.region
KB_NAME     = args.kb_name
VECTOR_COLLECTION_NAME = f"{KB_NAME}-vectors"


def wait_for(msg: str, check_fn, timeout: int = 300, poll: int = 10):
    """Poll until check_fn() returns truthy or timeout."""
    print(f"  ⏳ {msg}", end="", flush=True)
    elapsed = 0
    while elapsed < timeout:
        result = check_fn()
        if result:
            print(" ✓")
            return result
        print(".", end="", flush=True)
        time.sleep(poll)
        elapsed += poll
    raise TimeoutError(f"Timed out waiting for: {msg}")


# ── Step 1: IAM Role for Bedrock KB ───────────────────────────────────────────
def create_kb_iam_role() -> str:
    role_name = f"{KB_NAME}-role"
    print(f"\n[1/5] Creating IAM role: {role_name}")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": ACCOUNT_ID},
                "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:knowledge-base/*"},
            },
        }],
    }

    kb_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{args.s3_bucket}",
                    f"arn:aws:s3:::{args.s3_bucket}/*",
                ],
            },
            {
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel"],
                "Resource": f"arn:aws:bedrock:{REGION}::foundation-model/{args.embedding_model}",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "aoss:APIAccessAll",
                    "aoss:DashboardsAccessAll",
                ],
                "Resource": f"arn:aws:aoss:{REGION}:{ACCOUNT_ID}:collection/*",
            },
        ],
    }

    try:
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Bedrock Knowledge Base service role",
        )
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="BedrockKBPolicy",
            PolicyDocument=json.dumps(kb_policy),
        )
        role_arn = role["Role"]["Arn"]
        print(f"  ✓ Role ARN: {role_arn}")
        time.sleep(10)  # IAM propagation delay
        return role_arn
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
            print(f"  ↩ Role already exists: {role_arn}")
            return role_arn
        raise


# ── Step 2: OpenSearch Serverless Collection ──────────────────────────────────
def create_oss_collection(kb_role_arn: str) -> str:
    print(f"\n[2/5] Creating OpenSearch Serverless collection: {VECTOR_COLLECTION_NAME}")

    # Encryption policy
    try:
        oss.create_security_policy(
            name=f"{VECTOR_COLLECTION_NAME}-enc",
            type="encryption",
            policy=json.dumps({
                "Rules": [{"ResourceType": "collection", "Resource": [f"collection/{VECTOR_COLLECTION_NAME}"]}],
                "AWSOwnedKey": True,
            }),
        )
    except ClientError as e:
        if "ConflictException" not in str(e): raise

    # Network policy (private — recommended for production)
    try:
        oss.create_security_policy(
            name=f"{VECTOR_COLLECTION_NAME}-net",
            type="network",
            policy=json.dumps([{
                "Rules": [
                    {"ResourceType": "collection", "Resource": [f"collection/{VECTOR_COLLECTION_NAME}"]},
                    {"ResourceType": "dashboard",  "Resource": [f"collection/{VECTOR_COLLECTION_NAME}"]},
                ],
                "AllowFromPublic": True,  # Change to False + VPC for production
            }]),
        )
    except ClientError as e:
        if "ConflictException" not in str(e): raise

    # Data access policy
    caller_arn = f"arn:aws:iam::{ACCOUNT_ID}:root"
    try:
        oss.create_access_policy(
            name=f"{VECTOR_COLLECTION_NAME}-access",
            type="data",
            policy=json.dumps([{
                "Rules": [
                    {"ResourceType": "collection", "Resource": [f"collection/{VECTOR_COLLECTION_NAME}"],
                     "Permission": ["aoss:CreateCollectionItems", "aoss:DescribeCollectionItems", "aoss:DeleteCollectionItems", "aoss:UpdateCollectionItems"]},
                    {"ResourceType": "index", "Resource": [f"index/{VECTOR_COLLECTION_NAME}/*"],
                     "Permission": ["aoss:CreateIndex", "aoss:DescribeIndex", "aoss:ReadDocument", "aoss:WriteDocument", "aoss:UpdateIndex", "aoss:DeleteIndex"]},
                ],
                "Principal": [caller_arn, kb_role_arn],
            }]),
        )
    except ClientError as e:
        if "ConflictException" not in str(e): raise

    # Create collection
    try:
        response = oss.create_collection(
            name=VECTOR_COLLECTION_NAME,
            type="VECTORSEARCH",
            description="Vector store for Bedrock Knowledge Base",
        )
        collection_id = response["createCollectionDetail"]["id"]
    except ClientError as e:
        if "OcuLimitExceededException" in str(e) or "ConflictException" in str(e):
            collections = oss.list_collections(collectionFilters={"name": VECTOR_COLLECTION_NAME})
            collection_id = collections["collectionSummaries"][0]["id"]
            print(f"  ↩ Collection already exists: {collection_id}")
        else:
            raise

    # Wait for ACTIVE
    collection_endpoint = wait_for(
        "Collection becoming active",
        lambda: next(
            (c["collectionEndpoint"] for c in oss.list_collections(
                collectionFilters={"name": VECTOR_COLLECTION_NAME}
            ).get("collectionSummaries", []) if c.get("status") == "ACTIVE"),
            None
        ),
        timeout=600,
        poll=15,
    )

    print(f"  ✓ Endpoint: {collection_endpoint}")
    return collection_endpoint


# ── Step 3: Create Bedrock Knowledge Base ─────────────────────────────────────
def create_knowledge_base(kb_role_arn: str, collection_endpoint: str) -> tuple[str, str]:
    print(f"\n[3/5] Creating Bedrock Knowledge Base: {KB_NAME}")

    try:
        response = bedrock_agent.create_knowledge_base(
            name=KB_NAME,
            description="Enterprise document Q&A knowledge base",
            roleArn=kb_role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": f"arn:aws:bedrock:{REGION}::foundation-model/{args.embedding_model}",
                    "embeddingModelConfiguration": {
                        "bedrockEmbeddingModelConfiguration": {
                            "dimensions": 1024,  # Titan Embed v2 default
                        }
                    },
                },
            },
            storageConfiguration={
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": f"arn:aws:aoss:{REGION}:{ACCOUNT_ID}:collection/{VECTOR_COLLECTION_NAME}",
                    "vectorIndexName": "bedrock-knowledge-base-index",
                    "fieldMapping": {
                        "vectorField": "bedrock-knowledge-base-default-vector",
                        "textField": "AMAZON_BEDROCK_TEXT_CHUNK",
                        "metadataField": "AMAZON_BEDROCK_METADATA",
                    },
                },
            },
        )
        kb_id = response["knowledgeBase"]["knowledgeBaseId"]
        kb_arn = response["knowledgeBase"]["knowledgeBaseArn"]
    except ClientError as e:
        if "ConflictException" in str(e):
            kbs = bedrock_agent.list_knowledge_bases()
            kb = next(k for k in kbs["knowledgeBaseSummaries"] if k["name"] == KB_NAME)
            kb_id = kb["knowledgeBaseId"]
            kb_arn = kb["knowledgeBaseArn"]
            print(f"  ↩ KB already exists: {kb_id}")
            return kb_id, kb_arn
        raise

    wait_for(
        "Knowledge Base becoming ACTIVE",
        lambda: bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)
                              ["knowledgeBase"]["status"] == "ACTIVE"
    )
    print(f"  ✓ Knowledge Base ID: {kb_id}")
    return kb_id, kb_arn


# ── Step 4: Create Data Source ─────────────────────────────────────────────────
def create_data_source(kb_id: str) -> str:
    print(f"\n[4/5] Creating S3 data source")

    response = bedrock_agent.create_data_source(
        knowledgeBaseId=kb_id,
        name="s3-documents",
        description="Company documents from S3",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": f"arn:aws:s3:::{args.s3_bucket}",
                "inclusionPrefixes": [args.s3_prefix],
            },
        },
        vectorIngestionConfiguration={
            "chunkingConfiguration": {
                "chunkingStrategy": "HIERARCHICAL",
                "hierarchicalChunkingConfiguration": {
                    "levelConfigurations": [
                        {"maxTokens": 1500},  # Parent chunks
                        {"maxTokens": 300},   # Child chunks (retrieved)
                    ],
                    "overlapTokens": 60,
                },
            },
            "parsingConfiguration": {
                "parsingStrategy": "BEDROCK_FOUNDATION_MODEL",
                "bedrockFoundationModelConfiguration": {
                    "modelArn": f"arn:aws:bedrock:{REGION}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                    "parsingPrompt": {
                        "parsingPromptText": (
                            "Extract all text content from this document. "
                            "Preserve headings, tables, and lists. "
                            "For tables, convert to descriptive text."
                        )
                    },
                },
            },
        },
    )

    ds_id = response["dataSource"]["dataSourceId"]
    print(f"  ✓ Data Source ID: {ds_id}")
    return ds_id


# ── Step 5: Trigger Initial Ingestion ─────────────────────────────────────────
def trigger_ingestion(kb_id: str, ds_id: str):
    print(f"\n[5/5] Starting document ingestion")

    response = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
        description="Initial document ingestion",
    )

    job_id = response["ingestionJob"]["ingestionJobId"]
    print(f"  📥 Ingestion job started: {job_id}")

    def check_complete():
        job = bedrock_agent.get_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            ingestionJobId=job_id,
        )["ingestionJob"]
        status = job["status"]
        stats = job.get("statistics", {})
        if status == "COMPLETE":
            print(f"\n  ✓ Indexed {stats.get('numberOfDocumentsScanned', 0)} documents, "
                  f"{stats.get('numberOfNewDocumentsIndexed', 0)} chunks created")
            return True
        elif status == "FAILED":
            raise RuntimeError(f"Ingestion failed: {job.get('failureReasons', [])}")
        return False

    wait_for("Ingestion completing", check_complete, timeout=1800, poll=20)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Enterprise KB Q&A — Knowledge Base Setup")
    print("=" * 60)
    print(f"  S3 Bucket:  {args.s3_bucket}/{args.s3_prefix}")
    print(f"  Region:     {REGION}")
    print(f"  Embedding:  {args.embedding_model}")
    print("=" * 60)

    try:
        kb_role_arn          = create_kb_iam_role()
        collection_endpoint  = create_oss_collection(kb_role_arn)
        kb_id, kb_arn        = create_knowledge_base(kb_role_arn, collection_endpoint)
        ds_id                = create_data_source(kb_id)
        trigger_ingestion(kb_id, ds_id)

        print("\n" + "=" * 60)
        print("  ✅ Setup Complete!")
        print("=" * 60)
        print(f"\n  Add this to your .env file:")
        print(f"  BEDROCK_KB_ID={kb_id}")
        print(f"\n  Knowledge Base ARN: {kb_arn}")
        print()

    except Exception as e:
        print(f"\n❌ Setup failed: {e}", file=sys.stderr)
        sys.exit(1)
