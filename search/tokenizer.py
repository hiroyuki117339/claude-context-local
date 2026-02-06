"""Tokenizer for BM25 keyword search.

Handles CJK bigrams, CamelCase/snake_case splitting, and basic stop word removal.
No external NLP dependencies (no MeCab, no spaCy).
"""

import re
from typing import List

# Minimal English stop words — just the most common function words
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "between", "through", "after", "before",
    "and", "or", "but", "not", "no", "if", "then", "so",
})

# Regex: CJK Unified Ideographs + CJK Extension A/B + common CJK ranges
_CJK_RE = re.compile(
    r'[\u3400-\u4DBF'   # CJK Unified Ideographs Extension A
    r'\u4E00-\u9FFF'    # CJK Unified Ideographs
    r'\uF900-\uFAFF'    # CJK Compatibility Ideographs
    r'\U00020000-\U0002A6DF'  # CJK Extension B
    r']'
)

# Hiragana and Katakana
_KANA_RE = re.compile(
    r'[\u3040-\u309F'   # Hiragana
    r'\u30A0-\u30FF'    # Katakana
    r']'
)

# CJK + Kana combined
_CJK_KANA_RE = re.compile(
    r'[\u3040-\u309F'   # Hiragana
    r'\u30A0-\u30FF'    # Katakana
    r'\u3400-\u4DBF'    # CJK Extension A
    r'\u4E00-\u9FFF'    # CJK Unified
    r'\uF900-\uFAFF'    # CJK Compatibility
    r'\U00020000-\U0002A6DF'  # CJK Extension B
    r']+'
)

# ASCII word pattern (letters, digits, underscores)
_ASCII_WORD_RE = re.compile(r'[A-Za-z0-9_]+')

# CamelCase split: insert space before uppercase preceded by lowercase
_CAMEL_RE = re.compile(r'([a-z0-9])([A-Z])')

# Consecutive uppercase followed by lowercase (e.g., "HTMLParser" → "HTML Parser")
_UPPER_CAMEL_RE = re.compile(r'([A-Z]+)([A-Z][a-z])')


def _extract_cjk_tokens(text: str) -> List[str]:
    """Extract CJK/Kana tokens using bigram + unigram."""
    tokens = []
    for run in _CJK_KANA_RE.findall(text):
        # Unigrams
        for ch in run:
            tokens.append(ch)
        # Bigrams
        for i in range(len(run) - 1):
            tokens.append(run[i:i + 2])
    return tokens


def _split_camel_and_snake(word: str) -> List[str]:
    """Split a word on CamelCase and snake_case boundaries."""
    # CamelCase split
    word = _CAMEL_RE.sub(r'\1 \2', word)
    word = _UPPER_CAMEL_RE.sub(r'\1 \2', word)
    # snake_case split
    parts = word.replace('_', ' ').replace('-', ' ').split()
    return [p.lower() for p in parts if p]


def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 indexing/querying.

    - CJK/Kana characters: unigram + bigram
    - ASCII/code tokens: CamelCase split, snake_case split, lowercased
    - Single-char ASCII tokens and stop words removed

    Args:
        text: Input text (code, natural language, or mixed).

    Returns:
        List of string tokens.
    """
    if not text:
        return []

    tokens: List[str] = []

    # 1) Extract CJK/Kana tokens (bigram + unigram)
    tokens.extend(_extract_cjk_tokens(text))

    # 2) Extract and process ASCII words
    for word in _ASCII_WORD_RE.findall(text):
        parts = _split_camel_and_snake(word)
        for part in parts:
            if len(part) <= 1:
                continue
            if part in _STOP_WORDS:
                continue
            tokens.append(part)

    return tokens
