#!/usr/bin/env python3
"""
Document Upload & Re-ingestion Script
=====================================
Uploads local documents to S3 and triggers Bedrock KB re-ingestion.

Supported formats: PDF, DOCX, TXT, HTML, CSV, XLSX, MD

Usage:
  python upload_documents.py --dir ./my-docs --bucket my-docs-bucket
  python upload_documents.py --file report.pdf --bucket my-docs-bucket
  python upload_documents.py --sync --bucket my-docs-bucket  # Re-sync KB only
"""

import boto3
import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".csv", ".xlsx", ".md"}

parser = argparse.ArgumentParser()
parser.add_argument("--bucket",   required=True)
parser.add_argument("--prefix",   default="documents/")
parser.add_argument("--dir",      help="Upload all supported files in directory")
parser.add_argument("--file",     help="Upload a single file")
parser.add_argument("--sync",     action="store_true", help="Trigger re-ingestion only")
parser.add_argument("--kb-id",    default=os.getenv("BEDROCK_KB_ID"))
parser.add_argument("--ds-id",    help="Data Source ID (auto-detected if not provided)")
parser.add_argument("--region",   default=os.getenv("AWS_REGION", "us-east-1"))
args = parser.parse_args()

s3     = boto3.client("s3", region_name=args.region)
agent  = boto3.client("bedrock-agent", region_name=args.region)


def upload_file(local_path: Path) -> str:
    """Upload a file to S3 and return the S3 key."""
    s3_key = f"{args.prefix}{local_path.name}"
    print(f"  ↑ Uploading: {local_path.name} → s3://{args.bucket}/{s3_key}")
    s3.upload_file(
        str(local_path), args.bucket, s3_key,
        ExtraArgs={"ServerSideEncryption": "AES256"},
    )
    return s3_key


def get_data_source_id(kb_id: str) -> str:
    """Auto-detect the first data source for the knowledge base."""
    sources = agent.list_data_sources(knowledgeBaseId=kb_id)["dataSourceSummaries"]
    if not sources:
        raise RuntimeError(f"No data sources found for KB: {kb_id}")
    return sources[0]["dataSourceId"]


def trigger_ingestion(kb_id: str, ds_id: str) -> str:
    """Start a sync job and return the job ID."""
    response = agent.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
        description=f"Re-sync triggered at {datetime.now().isoformat()}",
    )
    return response["ingestionJob"]["ingestionJobId"]


def wait_for_ingestion(kb_id: str, ds_id: str, job_id: str):
    print(f"\n  ⏳ Waiting for ingestion job {job_id}", end="", flush=True)
    while True:
        job = agent.get_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            ingestionJobId=job_id,
        )["ingestionJob"]
        status = job["status"]

        if status == "COMPLETE":
            stats = job.get("statistics", {})
            print(f" ✓")
            print(f"\n  📊 Results:")
            print(f"     Scanned:   {stats.get('numberOfDocumentsScanned', 0)}")
            print(f"     Indexed:   {stats.get('numberOfNewDocumentsIndexed', 0)}")
            print(f"     Modified:  {stats.get('numberOfModifiedDocumentsIndexed', 0)}")
            print(f"     Deleted:   {stats.get('numberOfDocumentsDeleted', 0)}")
            print(f"     Failed:    {stats.get('numberOfDocumentsFailed', 0)}")
            return
        elif status == "FAILED":
            reasons = job.get("failureReasons", ["Unknown error"])
            raise RuntimeError(f"Ingestion failed: {reasons}")
        elif status in ("STARTING", "IN_PROGRESS"):
            print(".", end="", flush=True)
            time.sleep(10)
        else:
            print(f"\n  ⚠ Unexpected status: {status}")
            time.sleep(10)


if __name__ == "__main__":
    uploaded = []

    if not args.sync:
        if args.file:
            path = Path(args.file)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                print(f"❌ Unsupported file type: {path.suffix}")
                sys.exit(1)
            upload_file(path)
            uploaded.append(path.name)

        elif args.dir:
            directory = Path(args.dir)
            files = [f for f in directory.iterdir()
                     if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

            if not files:
                print(f"⚠ No supported files found in {directory}")
                sys.exit(0)

            print(f"\n📤 Uploading {len(files)} files to s3://{args.bucket}/{args.prefix}")
            for f in sorted(files):
                upload_file(f)
                uploaded.append(f.name)
        else:
            print("❌ Provide --file, --dir, or --sync")
            sys.exit(1)

    # ── Trigger re-ingestion ────────────────────────────────────────────────
    if args.kb_id:
        ds_id = args.ds_id or get_data_source_id(args.kb_id)
        print(f"\n🔄 Triggering KB re-ingestion (KB: {args.kb_id}, DS: {ds_id})")
        job_id = trigger_ingestion(args.kb_id, ds_id)
        wait_for_ingestion(args.kb_id, ds_id, job_id)
        print("\n✅ Documents available in Knowledge Base!")
    else:
        print(f"\n✅ Uploaded {len(uploaded)} files. Set BEDROCK_KB_ID to trigger ingestion.")
