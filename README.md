# Enterprise Knowledge Base Q&A System
## Powered by Amazon Bedrock Knowledge Bases (RAG)

A production-ready Retrieval-Augmented Generation (RAG) application that enables employees to ask natural language questions about proprietary company documents and receive **accurate, citation-backed answers** — without hallucination.

---

## Architecture Overview

```
Employee Browser
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  EC2 Instance (t3.medium)  ·  IAM Role (no hardcoded keys)  │
│                                                             │
│  ┌─────────────────────────────────────┐                   │
│  │      Streamlit Web UI (port 8501)   │                   │
│  │      app/main.py                    │                   │
│  └──────────────────┬──────────────────┘                   │
│                     │                                       │
│  ┌──────────────────▼──────────────────┐                   │
│  │      BedrockKBClient                │                   │
│  │      bedrock-agent-runtime API      │                   │
│  └──────────────────┬──────────────────┘                   │
└─────────────────────│───────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │   Amazon Bedrock      │
          │   Knowledge Bases     │
          │                       │
          │  1. Embed query       │
          │     (Titan Embed v2)  │
          │                       │
          │  2. Vector search     │
          │     (OpenSearch       │
          │      Serverless)      │
          │                       │
          │  3. Generate answer   │
          │     (Claude 3 Sonnet) │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │  S3 Document Store    │
          │  (PDFs, DOCX, TXT…)   │
          └───────────────────────┘
```

### RAG Flow (Step by Step)

1. **User submits query** via Streamlit UI
2. **Query embedding**: Bedrock converts query to 1024-dim vector using Titan Embeddings v2
3. **Semantic search**: Vector similarity search against OpenSearch Serverless finds the top-K most relevant document chunks
4. **Context assembly**: Retrieved chunks are injected into the prompt as grounded context
5. **Generation**: Claude 3 Sonnet generates an answer **strictly from the retrieved context** (no hallucination)
6. **Citation rendering**: Source documents and relevance scores are displayed alongside the answer

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| AWS CLI | 2.x |
| Terraform | 1.5+ (optional) |
| AWS Account | Bedrock enabled in region |

### Required AWS Permissions

Your IAM user/role needs these to run setup:
- `bedrock:*` (or scoped to knowledge-base actions)
- `s3:CreateBucket`, `s3:PutObject`, `s3:GetObject`
- `iam:CreateRole`, `iam:PutRolePolicy`
- `aoss:*` (OpenSearch Serverless)

---

## Quick Start

### 1. Upload Documents to S3

```bash
# Create your S3 bucket
aws s3 mb s3://my-company-docs-bucket --region us-east-1

# Upload documents (PDF, DOCX, TXT, HTML, CSV, XLSX supported)
python scripts/upload_documents.py \
    --dir ./my-internal-docs \
    --bucket my-company-docs-bucket
```

### 2. Create the Knowledge Base

```bash
pip install boto3

python scripts/setup_knowledge_base.py \
    --s3-bucket my-company-docs-bucket \
    --s3-prefix documents/ \
    --region us-east-1

# Output: BEDROCK_KB_ID=XXXXXXXXXX  ← copy this!
```

### 3. Configure Environment

```bash
# app/.env
BEDROCK_KB_ID=XXXXXXXXXX          # From step 2
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
DEFAULT_NUM_RESULTS=5
DEFAULT_MIN_SCORE=0.3
```

### 4. Run Locally

```bash
cd app
pip install -r requirements.txt
streamlit run main.py
# → http://localhost:8501
```

### 5. Deploy to EC2 (Terraform)

```bash
cd infrastructure

terraform init
terraform apply \
    -var="knowledge_base_id=XXXXXXXXXX" \
    -var="aws_region=us-east-1" \
    -var="allowed_cidr=203.0.113.0/24"  # Your office IP

# Output: app_url = "http://1.2.3.4:8501"
```

---

## Project Structure

```
enterprise-kb-rag/
├── app/
│   ├── main.py              # Streamlit UI
│   ├── bedrock_client.py    # Bedrock KB API wrapper
│   ├── config.py            # Environment config
│   ├── utils.py             # Citation rendering, input sanitization
│   ├── requirements.txt
│   └── assets/
│       └── styles.css       # Custom dark theme UI
│
├── infrastructure/
│   ├── main.tf              # Terraform: EC2, IAM, S3, Security Group
│   └── user_data.sh         # EC2 bootstrap (installs app as systemd service)
│
├── scripts/
│   ├── setup_knowledge_base.py   # One-time KB creation
│   └── upload_documents.py       # Upload docs + trigger re-ingestion
│
└── tests/
    └── test_app.py          # Unit tests (pytest)
```

---

## Key Design Decisions

### Why Amazon Bedrock Knowledge Bases?
- **Fully managed**: No vector DB operations, no embedding pipeline maintenance
- **Hybrid search**: Combines semantic + keyword search for higher recall
- **Native citations**: The API returns source references with every answer
- **IAM-native security**: No API keys to manage; works with EC2 instance profiles

### Why Hierarchical Chunking?
Configured in `setup_knowledge_base.py`:
- **Parent chunks (1500 tokens)**: Preserve document context
- **Child chunks (300 tokens)**: Fine-grained retrieval precision
- The model sees parent context but searches via child vectors — best of both worlds

### Why Claude 3 Sonnet?
Optimal cost/quality for enterprise Q&A. Swap to:
- **Claude 3 Haiku**: 3× cheaper, slightly lower quality — good for high-volume
- **Claude 3 Opus**: Best quality, 5× more expensive — use for complex analysis

### IAM Best Practices
- EC2 uses an **instance profile** (no hardcoded credentials)
- KB service role follows **least privilege** (S3 read-only, specific KB actions)
- No `*` on sensitive resource ARNs in production

---

## Adding New Documents

```bash
# Upload and auto-sync
python scripts/upload_documents.py \
    --file new-policy-2024.pdf \
    --bucket my-company-docs-bucket \
    --kb-id YOUR_KB_ID

# Re-sync after bulk upload
python scripts/upload_documents.py \
    --sync \
    --bucket my-company-docs-bucket \
    --kb-id YOUR_KB_ID
```

---

## Running Tests

```bash
cd enterprise-kb-rag
pip install pytest
pytest tests/ -v --tb=short
```

---

## Production Hardening Checklist

- [ ] Restrict `allowed_cidr` to corporate IP range in Terraform
- [ ] Set OpenSearch Serverless network policy to VPC-only
- [ ] Enable CloudTrail for Bedrock API audit logging
- [ ] Add Application Load Balancer + HTTPS (TLS termination)
- [ ] Configure Cognito or SSO for user authentication
- [ ] Set up CloudWatch alarms on Bedrock throttling metrics
- [ ] Schedule daily ingestion jobs via EventBridge
- [ ] Tag all resources for cost allocation

---

## Cost Estimation (us-east-1, moderate usage)

| Component | Cost |
|---|---|
| EC2 t3.medium | ~$30/mo |
| OpenSearch Serverless (0.5 OCU) | ~$175/mo |
| Titan Embeddings v2 (ingestion) | ~$0.02/1K tokens |
| Claude 3 Sonnet (Q&A queries) | ~$3/1M input tokens |
| S3 (documents + vectors) | < $5/mo |

> Tip: Use Claude 3 Haiku for cost-sensitive deployments (~10× cheaper for generation).
