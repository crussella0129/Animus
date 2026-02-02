"""Token counting utilities using tiktoken.

This module provides accurate token counting for LLM context management.
Uses tiktoken (OpenAI's tokenizer) which is widely compatible with most models.
"""

from __future__ import annotations

from typing import Optional, Union
from functools import lru_cache

# Global state for lazy loading
_TIKTOKEN_AVAILABLE: Optional[bool] = None
_ENCODING = None


def _check_tiktoken() -> bool:
    """Check if tiktoken is available and cache the result."""
    global _TIKTOKEN_AVAILABLE

    if _TIKTOKEN_AVAILABLE is not None:
        return _TIKTOKEN_AVAILABLE

    try:
        import tiktoken  # noqa: F401
        _TIKTOKEN_AVAILABLE = True
    except ImportError:
        _TIKTOKEN_AVAILABLE = False

    return _TIKTOKEN_AVAILABLE


def _get_encoding():
    """Get the tiktoken encoding, initializing if needed.

    Uses cl100k_base encoding which is used by GPT-4, GPT-3.5-turbo,
    and is a good general-purpose tokenizer for most LLMs.
    """
    global _ENCODING

    if _ENCODING is not None:
        return _ENCODING

    if not _check_tiktoken():
        return None

    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken.

    Uses cl100k_base encoding for accurate token counting.
    Falls back to character-based estimation if tiktoken is unavailable.

    Args:
        text: Text to count tokens for.

    Returns:
        Token count.
    """
    if not text:
        return 0

    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))

    # Fallback: rough estimation based on characters
    # Average is ~4 chars per token for English text
    return len(text) // 4


def count_tokens_messages(messages: list[dict]) -> int:
    """Count tokens for a list of chat messages.

    Accounts for message overhead (role, formatting, etc.)

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        Total token count including overhead.
    """
    total = 0
    for message in messages:
        # Each message has ~4 tokens of overhead for role and formatting
        total += 4
        if isinstance(message, dict):
            total += count_tokens(message.get("content", ""))
            total += count_tokens(message.get("role", ""))
            if message.get("name"):
                total += count_tokens(message.get("name", ""))
                total += 1  # name overhead
        else:
            # Handle Message objects
            total += count_tokens(getattr(message, "content", ""))
            total += count_tokens(getattr(message, "role", ""))

    # Add 2 tokens for assistant reply priming
    total += 2

    return total


def estimate_tokens_rough(text: str) -> int:
    """Rough token estimation without tiktoken.

    Uses character-based heuristics:
    - ~4 characters per token for English text
    - ~3 characters per token for code (more symbols)
    - ~6 characters per token for non-Latin scripts

    This is a fallback when tiktoken is unavailable.

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    # Check for code indicators
    code_indicators = ['{', '}', '(', ')', '[', ']', ';', '//', '/*', '#include', 'def ', 'class ', 'function ']
    is_code = any(indicator in text for indicator in code_indicators)

    if is_code:
        # Code tends to have more tokens per character
        return len(text) // 3
    else:
        # Standard English text
        return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int, suffix: str = "...") -> str:
    """Truncate text to a maximum number of tokens.

    Args:
        text: Text to truncate.
        max_tokens: Maximum tokens to allow.
        suffix: Suffix to add if truncated.

    Returns:
        Truncated text.
    """
    if not text or max_tokens <= 0:
        return ""

    encoding = _get_encoding()
    if encoding is None:
        # Fallback: estimate based on characters
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars - len(suffix)] + suffix

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    # Reserve space for suffix
    suffix_tokens = len(encoding.encode(suffix))
    truncated_tokens = tokens[:max_tokens - suffix_tokens]

    return encoding.decode(truncated_tokens) + suffix


def split_by_tokens(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split text into chunks of approximately chunk_size tokens.

    Args:
        text: Text to split.
        chunk_size: Target tokens per chunk.
        overlap: Number of tokens to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if not text or chunk_size <= 0:
        return []

    encoding = _get_encoding()
    if encoding is None:
        # Fallback: split by characters
        char_chunk = chunk_size * 4
        char_overlap = overlap * 4
        chunks = []
        i = 0
        while i < len(text):
            end = min(i + char_chunk, len(text))
            chunks.append(text[i:end])
            i += char_chunk - char_overlap
        return chunks

    tokens = encoding.encode(text)
    chunks = []
    i = 0

    while i < len(tokens):
        end = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:end]
        chunks.append(encoding.decode(chunk_tokens))
        i += chunk_size - overlap

    return chunks


@lru_cache(maxsize=1000)
def count_tokens_cached(text: str) -> int:
    """Count tokens with caching for repeated text.

    Use this for text that may be counted multiple times.

    Args:
        text: Text to count tokens for.

    Returns:
        Token count.
    """
    return count_tokens(text)


def is_tiktoken_available() -> bool:
    """Check if tiktoken is available.

    Returns:
        True if tiktoken can be used, False otherwise.
    """
    return _check_tiktoken()


def get_model_context_limit(model_name: str) -> int:
    """Get the context window size for a model.

    Args:
        model_name: Name of the model.

    Returns:
        Context window size in tokens.
    """
    model_lower = model_name.lower()

    # Common model context limits
    # Order matters - check more specific patterns first
    context_limits = [
        # LLaMA 3 (check before generic llama)
        ("llama-3", 8192),
        ("llama3", 8192),
        # LLaMA 2
        ("llama-2", 4096),
        ("llama2", 4096),
        # Generic LLaMA
        ("llama", 4096),
        # Qwen models
        ("qwen2.5", 32768),
        ("qwen2", 32768),
        ("qwen", 32768),
        # Mistral/Mixtral
        ("mixtral", 32768),
        ("mistral", 8192),
        # Code models
        ("codellama", 16384),
        ("deepseek", 16384),
        ("starcoder", 8192),
        # Phi models
        ("phi-3", 4096),
        ("phi-2", 2048),
        ("phi", 2048),
        # TinyLlama
        ("tinyllama", 2048),
    ]

    for name, limit in context_limits:
        if name in model_lower:
            return limit

    # Default context limit
    return 4096
