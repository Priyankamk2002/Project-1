"""
Enterprise Knowledge Base Q&A System
Powered by Amazon Bedrock Knowledge Bases (RAG)
"""

import streamlit as st
import boto3
import json
import time
import uuid
from datetime import datetime
from typing import Optional
from config import AppConfig
from bedrock_client import BedrockKBClient
from utils import format_citations, render_confidence_badge, sanitize_query

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Knowledge Base",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Custom CSS ──────────────────────────────────────────────────────────
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Initialize Session State ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "client" not in st.session_state:
    st.session_state.client = None

# ─── Initialize Bedrock Client ────────────────────────────────────────────────
@st.cache_resource
def get_bedrock_client():
    config = AppConfig()
    return BedrockKBClient(
        knowledge_base_id=config.KNOWLEDGE_BASE_ID,
        region_name=config.AWS_REGION,
        model_arn=config.MODEL_ARN,
    )

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">⬡</div>
        <div>
            <div class="sidebar-title">Enterprise KB</div>
            <div class="sidebar-subtitle">RAG-Powered Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Connection Status
    st.markdown("### ⚡ System Status")
    config = AppConfig()

    try:
        client = get_bedrock_client()
        st.session_state.client = client
        st.markdown("""
        <div class="status-badge status-ok">
            <span>●</span> Bedrock Connected
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown("""
        <div class="status-badge status-error">
            <span>●</span> Connection Failed
        </div>
        """, unsafe_allow_html=True)
        st.error(f"Error: {str(e)}")

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Retrieval Settings")

    num_results = st.slider(
        "Max source chunks",
        min_value=1, max_value=10, value=config.DEFAULT_NUM_RESULTS,
        help="Number of document chunks to retrieve per query"
    )

    min_score = st.slider(
        "Min relevance score",
        min_value=0.0, max_value=1.0, value=config.DEFAULT_MIN_SCORE, step=0.05,
        help="Filter out chunks below this relevance threshold"
    )

    search_type = st.selectbox(
        "Search strategy",
        options=["HYBRID", "SEMANTIC", "KEYWORD"],
        index=0,
        help="HYBRID combines semantic + keyword search for best results"
    )

    show_sources = st.toggle("Show source citations", value=True)
    show_scores = st.toggle("Show relevance scores", value=False)

    st.markdown("---")

    # Session Info
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.query_count)
    with col2:
        st.metric("Messages", len(st.session_state.messages))

    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    st.markdown("---")

    # Clear history
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    # KB Info
    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    st.info(f"**KB ID:** `{config.KNOWLEDGE_BASE_ID}`\n\n**Region:** `{config.AWS_REGION}`\n\n**Model:** Claude 3 Sonnet")

# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Enterprise Knowledge Base <span class="header-accent">Q&A</span></h1>
    <p class="header-desc">Ask questions about your company documents. Powered by Amazon Bedrock RAG.</p>
</div>
""", unsafe_allow_html=True)

# Suggested questions (shown when no messages)
if not st.session_state.messages:
    st.markdown("### 💡 Suggested Questions")
    suggested = [
        "What is our data retention policy?",
        "Summarize the Q3 financial highlights",
        "What are the employee onboarding steps?",
        "Explain our security compliance framework",
        "What are the approved cloud vendors?",
    ]

    cols = st.columns(len(suggested))
    for i, (col, suggestion) in enumerate(zip(cols, suggested)):
        with col:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state.pending_query = suggestion
                st.rerun()

    st.markdown("---")

# ─── Chat History ─────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="message user-message">
                <div class="message-avatar">👤</div>
                <div class="message-content">
                    <div class="message-text">{msg["content"]}</div>
                    <div class="message-meta">{msg.get("timestamp", "")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            citations_html = ""
            if show_sources and msg.get("citations"):
                citations_html = format_citations(msg["citations"], show_scores)

            st.markdown(f"""
            <div class="message assistant-message">
                <div class="message-avatar">🔍</div>
                <div class="message-content">
                    <div class="message-text">{msg["content"]}</div>
                    {citations_html}
                    <div class="message-meta">{msg.get("timestamp", "")} · {msg.get("latency_ms", 0)}ms · {msg.get("chunks_retrieved", 0)} chunks</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ─── Query Input ──────────────────────────────────────────────────────────────
st.markdown("---")

with st.container():
    col_input, col_btn = st.columns([5, 1])

    # Handle pending query from suggestion buttons
    pending = st.session_state.pop("pending_query", None)

    with col_input:
        query = st.text_input(
            "Ask a question",
            value=pending or "",
            placeholder="e.g. What is the company's remote work policy?",
            label_visibility="collapsed",
            key="query_input"
        )

    with col_btn:
        submit = st.button("Ask →", type="primary", use_container_width=True)

# ─── Process Query ────────────────────────────────────────────────────────────
if (submit or pending) and query.strip():
    sanitized = sanitize_query(query.strip())
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": sanitized,
        "timestamp": timestamp,
    })
    st.session_state.query_count += 1

    # Show spinner while retrieving
    with st.spinner("🔍 Searching knowledge base..."):
        start_time = time.time()
        try:
            client = get_bedrock_client()
            response = client.query(
                question=sanitized,
                num_results=num_results,
                min_score=min_score,
                search_type=search_type,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "citations": response.get("citations", []),
                "chunks_retrieved": response.get("chunks_retrieved", 0),
                "latency_ms": latency_ms,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ **Error retrieving answer:** {str(e)}\n\nPlease check your Bedrock Knowledge Base configuration.",
                "citations": [],
                "chunks_retrieved": 0,
                "latency_ms": 0,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

    st.rerun()
