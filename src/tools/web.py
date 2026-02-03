"""Web search and fetch tools with security isolation.

This module implements the **Ungabunga-Box Agent** security pattern:

    "Animus tell little Ungabunga to go get web data. Animus not know if data
    is safe, so Animus put Ungabunga in box with data. If contents of box bad,
    Animus smash box."

Security layers:
1. Process isolation (Ungabunga runs in subprocess with no credentials)
2. Content sanitization (HTML stripped, plain text only)
3. Rule-based validation (prompt injection detection)
4. Human escalation for suspicious content (Animus ask human: smash box?)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional
from urllib.parse import urlparse, quote_plus

from src.tools.base import Tool, ToolParameter, ToolResult, ToolCategory


# =============================================================================
# Prompt Injection Detection Patterns
# =============================================================================

INJECTION_PATTERNS = [
    # Direct instruction override
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|commands)", "instruction_override"),
    (r"disregard\s+(all\s+)?(previous|prior|above|earlier)", "instruction_override"),
    (r"forget\s+(everything|all|what)\s+(you|i)\s+(said|told|instructed|were\s+told)", "instruction_override"),
    (r"new\s+instructions?\s*:", "instruction_override"),

    # Role manipulation
    (r"you\s+are\s+(now|actually)\s+(a|an)", "role_manipulation"),
    (r"pretend\s+(to\s+be|you'?re)", "role_manipulation"),
    (r"act\s+as\s+(if|though)", "role_manipulation"),
    (r"your\s+(new|real)\s+(purpose|goal|instructions|role)", "role_manipulation"),
    (r"from\s+now\s+on\s*,?\s*you", "role_manipulation"),

    # System prompt extraction
    (r"(reveal|show|display|print|output|repeat)\s+(me\s+)?(your|the)\s+(system|initial|original)\s+(prompt|instructions)", "prompt_extraction"),
    (r"what\s+(are|were)\s+your\s+(original|initial|system)\s+instructions", "prompt_extraction"),

    # Tool/command manipulation
    (r"(run|execute|call|invoke)\s+(the\s+)?(shell|command|bash|terminal)", "command_injection"),
    (r"write\s+(to\s+)?(a\s+)?file", "command_injection"),
    (r"delete\s+(the\s+|all\s+)?files?", "command_injection"),
    (r"rm\s+-rf", "command_injection"),
    (r"sudo\s+", "command_injection"),

    # Data exfiltration
    (r"send\s+(this|the|my|your)\s+(data|information|content)\s+to", "exfiltration"),
    (r"(upload|post|transmit)\s+.*\s+to\s+http", "exfiltration"),

    # Encoding tricks (often used to hide payloads)
    (r"base64\s*[:\[]", "encoded_payload"),
    (r"eval\s*\(", "code_execution"),
    (r"exec\s*\(", "code_execution"),
]

# URLs that should never appear in content
SUSPICIOUS_URL_PATTERNS = [
    r"file://",
    r"javascript:",
    r"data:text/html",
    r"vbscript:",
]


@dataclass
class ValidationResult:
    """Result of content validation."""
    is_safe: bool
    confidence: float  # 0.0 to 1.0
    issues: list[str]
    requires_human_review: bool = False


def validate_content(content: str, original_query: str) -> ValidationResult:
    """
    Validate web content for safety using rule-based checks.

    Args:
        content: The sanitized text content to validate.
        original_query: The original search query (for relevance checking).

    Returns:
        ValidationResult with safety assessment.
    """
    issues = []
    content_lower = content.lower()

    # Check for prompt injection patterns
    for pattern, category in INJECTION_PATTERNS:
        if re.search(pattern, content_lower, re.IGNORECASE):
            issues.append(f"Potential {category}: matched pattern '{pattern}'")

    # Check for suspicious URLs
    for pattern in SUSPICIOUS_URL_PATTERNS:
        if re.search(pattern, content_lower, re.IGNORECASE):
            issues.append(f"Suspicious URL scheme: {pattern}")

    # Check for excessive special characters (possible obfuscation)
    special_char_ratio = len(re.findall(r'[^\w\s.,!?;:\'"()-]', content)) / max(len(content), 1)
    if special_char_ratio > 0.3:
        issues.append(f"High special character ratio ({special_char_ratio:.2%})")

    # Determine safety
    if not issues:
        return ValidationResult(is_safe=True, confidence=0.9, issues=[])
    elif len(issues) == 1 and "special character" in issues[0]:
        # Minor issue, but flag for review
        return ValidationResult(
            is_safe=True,
            confidence=0.7,
            issues=issues,
            requires_human_review=True
        )
    else:
        # Serious issues detected
        return ValidationResult(
            is_safe=False,
            confidence=0.3,
            issues=issues,
            requires_human_review=True
        )


def sanitize_html(html_content: str) -> str:
    """
    Sanitize HTML content to plain text.

    Strips all HTML tags, scripts, styles, and returns clean text.

    Args:
        html_content: Raw HTML content.

    Returns:
        Plain text with all HTML removed.
    """
    # Try to use readability + bleach for best results
    try:
        from readability import Document
        doc = Document(html_content)
        html_content = doc.summary()
    except ImportError:
        pass  # Readability not installed, use raw HTML

    try:
        import bleach
        # Strip ALL tags, keeping only text
        text = bleach.clean(
            html_content,
            tags=[],  # No tags allowed
            strip=True,
            strip_comments=True,
        )
    except ImportError:
        # Fallback: basic regex-based stripping
        text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Decode HTML entities
    try:
        import html
        text = html.unescape(text)
    except Exception:
        pass

    return text


# =============================================================================
# Isolated Fetch Service
# =============================================================================

FETCH_SCRIPT = '''
"""Isolated fetch script - runs in subprocess with no credentials."""
import sys
import json

def fetch_url(url: str, timeout: int = 10, max_size: int = 1_000_000) -> dict:
    """Fetch URL content with safety limits."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx not installed", "content": None}

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, max_redirects=5) as client:
            # First, check content-type and size with HEAD
            try:
                head_response = client.head(url)
                content_type = head_response.headers.get("content-type", "")
                content_length = int(head_response.headers.get("content-length", 0))

                # Only allow text content
                if not any(t in content_type.lower() for t in ["text/", "application/json", "application/xml"]):
                    return {"error": f"Unsupported content type: {content_type}", "content": None}

                if content_length > max_size:
                    return {"error": f"Content too large: {content_length} bytes", "content": None}
            except Exception:
                pass  # HEAD failed, try GET anyway

            # Fetch content
            response = client.get(url)
            response.raise_for_status()

            # Check actual size
            if len(response.content) > max_size:
                return {"error": f"Content too large: {len(response.content)} bytes", "content": None}

            return {
                "content": response.text,
                "url": str(response.url),
                "status": response.status_code,
                "content_type": response.headers.get("content-type", ""),
            }

    except httpx.TimeoutException:
        return {"error": "Request timed out", "content": None}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP error: {e.response.status_code}", "content": None}
    except Exception as e:
        return {"error": str(e), "content": None}


def search_duckduckgo(query: str) -> dict:
    """Search using DuckDuckGo Instant Answers API."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx not installed", "results": []}

    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        with httpx.Client(timeout=10) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()

            results = []

            # Abstract (main answer)
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", "Answer"),
                    "snippet": data.get("Abstract"),
                    "url": data.get("AbstractURL", ""),
                    "source": data.get("AbstractSource", "DuckDuckGo"),
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:100],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "source": "DuckDuckGo",
                    })

            return {"results": results, "error": None}

    except Exception as e:
        return {"error": str(e), "results": []}


if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    arg = sys.argv[2] if len(sys.argv) > 2 else ""

    if action == "search":
        result = search_duckduckgo(arg)
    elif action == "fetch":
        result = fetch_url(arg)
    else:
        result = {"error": f"Unknown action: {action}"}

    print(json.dumps(result))
'''


async def run_isolated_fetch(action: str, arg: str) -> dict:
    """
    Run fetch operation in isolated subprocess.

    The subprocess runs with:
    - No environment variables (credentials can't leak)
    - Limited to text content types
    - Size limits enforced
    - Timeout enforced

    Args:
        action: "search" or "fetch"
        arg: Search query or URL

    Returns:
        Dict with results or error.
    """
    # Create temporary script file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(FETCH_SCRIPT)
        script_path = f.name

    try:
        # Run in subprocess with NO environment variables
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            script_path,
            action,
            arg,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={},  # NO credentials leak
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            return {"error": "Subprocess timed out"}

        if proc.returncode != 0:
            return {"error": f"Subprocess failed: {stderr.decode()[:200]}"}

        try:
            return json.loads(stdout.decode())
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON response: {stdout.decode()[:200]}"}

    finally:
        # Clean up temp file
        try:
            os.unlink(script_path)
        except Exception:
            pass


# =============================================================================
# Web Search Tool
# =============================================================================

class WebSearchTool(Tool):
    """
    Search the web securely.

    This tool:
    1. Runs search in isolated subprocess (no credential leakage)
    2. Sanitizes results (HTML stripped to plain text)
    3. Validates content for prompt injection attempts
    4. Escalates to human for suspicious content
    """

    def __init__(
        self,
        confirm_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
    ):
        """
        Initialize web search tool.

        Args:
            confirm_callback: Callback to ask user for confirmation.
                             If None, suspicious content is auto-rejected.
        """
        self.confirm_callback = confirm_callback

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return """Search the web for information.

Returns search results with titles, snippets, and URLs.
Content is sanitized and validated for safety before being returned.

Use this when you need current information not in your training data."""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="The search query",
                required=True,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def requires_confirmation(self) -> bool:
        return False  # Search itself doesn't need confirmation

    async def execute(self, query: str, **kwargs: Any) -> ToolResult:
        """Execute web search."""
        if not query or not query.strip():
            return ToolResult(
                success=False,
                output="",
                error="Search query cannot be empty",
            )

        # Run search in isolated subprocess
        result = await run_isolated_fetch("search", query.strip())

        if result.get("error"):
            return ToolResult(
                success=False,
                output="",
                error=f"Search failed: {result['error']}",
            )

        results = result.get("results", [])
        if not results:
            return ToolResult(
                success=True,
                output="No results found.",
                data={"results": []},
            )

        # Validate and format results
        validated_results = []
        for r in results:
            snippet = r.get("snippet", "")

            # Validate content
            validation = validate_content(snippet, query)

            if not validation.is_safe:
                if validation.requires_human_review and self.confirm_callback:
                    # Ask user
                    preview = snippet[:200] + "..." if len(snippet) > 200 else snippet
                    message = (
                        f"Suspicious content detected in search result:\n"
                        f"Issues: {', '.join(validation.issues)}\n"
                        f"Preview: {preview}\n"
                        f"Allow this content?"
                    )
                    if not await self.confirm_callback(message):
                        continue  # Skip this result
                else:
                    continue  # Auto-reject

            validated_results.append({
                "title": r.get("title", ""),
                "snippet": snippet,
                "url": r.get("url", ""),
                "source": r.get("source", ""),
            })

        # Format output
        output_lines = []
        for i, r in enumerate(validated_results, 1):
            output_lines.append(f"## Result {i}: {r['title']}")
            output_lines.append(f"{r['snippet']}")
            if r['url']:
                output_lines.append(f"URL: {r['url']}")
            output_lines.append("")

        return ToolResult(
            success=True,
            output="\n".join(output_lines) if output_lines else "No safe results found.",
            data={"results": validated_results},
        )


class WebFetchTool(Tool):
    """
    Fetch content from a specific URL securely.

    This tool:
    1. Runs fetch in isolated subprocess
    2. Validates URL (no file://, javascript:, etc.)
    3. Sanitizes HTML to plain text
    4. Validates content for prompt injection
    5. Escalates to human for suspicious content
    """

    def __init__(
        self,
        confirm_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
        max_content_length: int = 10000,
    ):
        """
        Initialize web fetch tool.

        Args:
            confirm_callback: Callback to ask user for confirmation.
            max_content_length: Maximum characters to return.
        """
        self.confirm_callback = confirm_callback
        self.max_content_length = max_content_length

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return """Fetch content from a URL.

Returns the main text content of the page (HTML is stripped).
Content is sanitized and validated for safety.

Use this to read articles, documentation, or other web pages."""

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="string",
                description="The URL to fetch",
                required=True,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def requires_confirmation(self) -> bool:
        return True  # Fetching arbitrary URLs should require confirmation

    async def execute(self, url: str, **kwargs: Any) -> ToolResult:
        """Fetch URL content."""
        if not url or not url.strip():
            return ToolResult(
                success=False,
                output="",
                error="URL cannot be empty",
            )

        url = url.strip()

        # Validate URL scheme
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Invalid URL scheme: {parsed.scheme}. Only http/https allowed.",
                )
            if not parsed.netloc:
                return ToolResult(
                    success=False,
                    output="",
                    error="Invalid URL: no hostname",
                )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid URL: {e}",
            )

        # Fetch in isolated subprocess
        result = await run_isolated_fetch("fetch", url)

        if result.get("error"):
            return ToolResult(
                success=False,
                output="",
                error=f"Fetch failed: {result['error']}",
            )

        raw_content = result.get("content", "")
        if not raw_content:
            return ToolResult(
                success=False,
                output="",
                error="No content received",
            )

        # Sanitize HTML to plain text
        content = sanitize_html(raw_content)

        # Truncate if too long
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "\n\n[Content truncated]"

        # Validate content
        validation = validate_content(content, url)

        if not validation.is_safe or validation.requires_human_review:
            if self.confirm_callback:
                preview = content[:500] + "..." if len(content) > 500 else content
                message = (
                    f"Content from {url} may contain suspicious patterns:\n"
                    f"Issues: {', '.join(validation.issues) if validation.issues else 'Low confidence'}\n"
                    f"Preview:\n{preview}\n\n"
                    f"Allow this content?"
                )
                if not await self.confirm_callback(message):
                    return ToolResult(
                        success=False,
                        output="",
                        error="Content rejected by user due to safety concerns",
                    )
            elif not validation.is_safe:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Content rejected: {', '.join(validation.issues)}",
                )

        return ToolResult(
            success=True,
            output=content,
            data={
                "url": result.get("url", url),
                "content_type": result.get("content_type", ""),
                "length": len(content),
            },
        )


def create_web_tools(
    confirm_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
) -> list[Tool]:
    """
    Create web search and fetch tools.

    Args:
        confirm_callback: Callback for human confirmation of suspicious content.

    Returns:
        List of web tools.
    """
    return [
        WebSearchTool(confirm_callback=confirm_callback),
        WebFetchTool(confirm_callback=confirm_callback),
    ]
