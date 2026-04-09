"""
Utility Functions for the Enterprise KB Q&A App
"""

import re
import html


def sanitize_query(query: str, max_length: int = 1000) -> str:
    """
    Sanitize user input before sending to Bedrock.
    - Strip leading/trailing whitespace
    - Collapse internal whitespace
    - Enforce max length
    - Remove null bytes
    """
    query = query.replace("\x00", "")              # Remove null bytes
    query = re.sub(r"\s+", " ", query).strip()     # Normalize whitespace
    return query[:max_length]


def format_citations(citations: list[dict], show_scores: bool = False) -> str:
    """
    Render citations as an HTML accordion/detail block.
    """
    if not citations:
        return ""

    items_html = ""
    seen_sources = set()

    for i, citation in enumerate(citations, 1):
        source = html.escape(citation.get("source", "Unknown"))
        text = html.escape(citation.get("text", "")[:300])
        score = citation.get("score", 0.0)

        # Deduplicate by source filename
        if source in seen_sources:
            continue
        seen_sources.add(source)

        score_html = ""
        if show_scores:
            score_pct = int(score * 100)
            score_html = f'<span class="score-badge">{score_pct}%</span>'

        items_html += f"""
        <div class="citation-item">
            <div class="citation-header">
                <span class="citation-num">#{i}</span>
                <span class="citation-source">📄 {source}</span>
                {score_html}
            </div>
            <div class="citation-text">"{text}..."</div>
        </div>
        """

    return f"""
    <div class="citations-block">
        <div class="citations-title">📚 Sources ({len(seen_sources)})</div>
        {items_html}
    </div>
    """


def render_confidence_badge(score: float) -> str:
    """Return a colored badge based on relevance score."""
    if score >= 0.8:
        return f'<span class="badge badge-high">High ({int(score*100)}%)</span>'
    elif score >= 0.5:
        return f'<span class="badge badge-medium">Medium ({int(score*100)}%)</span>'
    else:
        return f'<span class="badge badge-low">Low ({int(score*100)}%)</span>'
