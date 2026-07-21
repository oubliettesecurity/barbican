"""Vendored content scanners (subset) for BARBICAN feature extraction.

Copied VERBATIM from oubliette_shield/scanners.py (the shield content-scanner
module) so BARBICAN ships with zero runtime dependency on oubliette_shield.
Only the two scanners BARBICAN's artifact features use are vendored:
``scan_invisible_text`` and ``scan_ai_generated`` (plus the ``ScanFinding``
dataclass and the module-level constants they reference). Behavior is identical
to the source module — do not simplify or re-tune; feature values must not drift.
"""

import dataclasses
import re

# ---------------------------------------------------------------------------
# ScanFinding dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ScanFinding:
    """A single finding from a content scanner."""

    scanner: str  # "secrets", "pii", "urls", "language", "gibberish", "refusal", "invisible_text"
    category: str  # Sub-type: "aws_key", "ssn", "phishing_url", etc.
    severity: str  # "critical", "high", "medium", "low", "info"
    text_match: str  # Matched text (redacted for PII)
    start: int  # Character offset in source text
    end: int  # Character offset end
    message: str  # Human-readable description


# ---------------------------------------------------------------------------
# Scanner 3: Invisible Text
# ---------------------------------------------------------------------------

_INVISIBLE_CHARS = {
    "​": ("zero_width_space", "high"),
    "‌": ("zero_width_non_joiner", "high"),
    "‍": ("zero_width_joiner", "high"),
    "﻿": ("byte_order_mark", "medium"),
    "‮": ("rtl_override", "high"),
    "‭": ("ltr_override", "high"),
    "­": ("soft_hyphen", "medium"),
    "⁠": ("word_joiner", "medium"),
}

# Cyrillic lookalikes for Latin characters
_CYRILLIC_LOOKALIKES = set("аеорсухіјһ")
_LATIN_CHARS = set("aeopscuxijh")


def scan_invisible_text(text: str) -> list[ScanFinding]:
    """Scan text for invisible Unicode characters and homoglyphs."""
    findings: list[ScanFinding] = []

    # Check for invisible characters
    for i, ch in enumerate(text):
        if ch in _INVISIBLE_CHARS:
            category, severity = _INVISIBLE_CHARS[ch]
            findings.append(
                ScanFinding(
                    scanner="invisible_text",
                    category=category,
                    severity=severity,
                    text_match=f"U+{ord(ch):04X}",
                    start=i,
                    end=i + 1,
                    message=f"Invisible character {category.replace('_', ' ')} at position {i}",
                )
            )

    # Check for homoglyph mixing (Cyrillic + Latin in same word)
    has_latin = False
    has_cyrillic = False
    for ch in text:
        if ch in _LATIN_CHARS:
            has_latin = True
        if ch in _CYRILLIC_LOOKALIKES:
            has_cyrillic = True
        if has_latin and has_cyrillic:
            break

    if has_latin and has_cyrillic:
        findings.append(
            ScanFinding(
                scanner="invisible_text",
                category="homoglyph",
                severity="high",
                text_match="mixed Latin/Cyrillic scripts",
                start=0,
                end=len(text),
                message=(
                    "Text contains mixed Latin and Cyrillic lookalike characters "
                    "(possible homoglyph attack)"
                ),
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Scanner 8: AI-Generated Text (opt-in)
# ---------------------------------------------------------------------------

_HEDGE_PHRASES = [
    "it is important to note",
    "it's worth noting",
    "it should be noted",
    "it is worth mentioning",
    "it's important to",
    "one might consider",
    "it is generally",
    "it is recommended",
    "it may be helpful",
    "it could be argued",
    "on the other hand",
    "in conclusion",
    "however, it is",
    "furthermore,",
    "additionally,",
    "moreover,",
    "nevertheless,",
    "in this context",
]


def scan_ai_generated(text: str, threshold: float = 0.65) -> list[ScanFinding]:
    """Heuristic scanner to fingerprint LLM-authored prompt injections.

    Uses five signals: sentence length uniformity, type-token ratio,
    hedging language frequency, repetitive phrasing, and vocabulary
    richness. This is opt-in and NOT included in scan_all() by default.

    Args:
        text: The text to analyze.
        threshold: Score above which to flag as likely AI-generated (0-1).

    Returns:
        List of ScanFinding (empty if below threshold).
    """
    if len(text) < 100:
        return []

    sentences = re.split(r"[.!?\n]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) < 3:
        return []

    words = text.lower().split()
    if len(words) < 20:
        return []

    scores = []

    # 1. Sentence length uniformity (LLMs produce uniform sentence lengths)
    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    if mean_len > 0:
        variance = sum((length - mean_len) ** 2 for length in lengths) / len(lengths)
        cv = (variance**0.5) / mean_len  # coefficient of variation
        # Low CV = very uniform = likely AI; humans have CV > 0.5 typically
        if cv < 0.25:
            scores.append(1.0)
        elif cv < 0.40:
            scores.append(0.6)
        else:
            scores.append(0.0)
    else:
        scores.append(0.0)

    # 2. Type-token ratio (TTR) - AI tends to have moderate TTR
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    # AI text: TTR typically 0.40-0.60; very high or low is more human
    if 0.35 <= ttr <= 0.55:
        scores.append(0.7)
    elif 0.30 <= ttr <= 0.65:
        scores.append(0.3)
    else:
        scores.append(0.0)

    # 3. Hedging language frequency
    text_lower = text.lower()
    hedge_count = sum(1 for p in _HEDGE_PHRASES if p in text_lower)
    hedge_density = hedge_count / len(sentences)
    if hedge_density > 0.3:
        scores.append(1.0)
    elif hedge_density > 0.15:
        scores.append(0.6)
    else:
        scores.append(0.0)

    # 4. Repetitive phrasing (bigram repetition rate)
    bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
    if bigrams:
        unique_bigrams = set(bigrams)
        bigram_repeat_rate = 1.0 - (len(unique_bigrams) / len(bigrams))
        if bigram_repeat_rate > 0.3:
            scores.append(0.8)
        elif bigram_repeat_rate > 0.15:
            scores.append(0.4)
        else:
            scores.append(0.0)
    else:
        scores.append(0.0)

    # 5. Vocabulary richness (hapax legomena ratio)
    word_freq: dict[str, int] = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    hapax = sum(1 for v in word_freq.values() if v == 1)
    hapax_ratio = hapax / len(words) if words else 0
    # AI tends to have lower hapax ratio (reuses vocabulary more)
    if hapax_ratio < 0.35:
        scores.append(0.7)
    elif hapax_ratio < 0.50:
        scores.append(0.3)
    else:
        scores.append(0.0)

    overall = sum(scores) / len(scores) if scores else 0

    if overall >= threshold:
        return [
            ScanFinding(
                scanner="ai_generated",
                category="llm_authored",
                severity="medium",
                text_match=text[:60] + "..." if len(text) > 60 else text,
                start=0,
                end=len(text),
                message=f"Text appears LLM-authored (score: {overall:.2f})",
            )
        ]

    return []
