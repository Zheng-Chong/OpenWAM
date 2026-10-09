"""Instruction-text patterns shared by the quality scan and the readers.

``episode_quality`` flags raw prompts that match these patterns (``bad_prompt``);
readers built with ``normalize_prompt: true`` rewrite them into one house style
before the text encoder sees them. One module keeps "what counts as an
artifact" and its cleanup in step.
"""

from __future__ import annotations

import re

# article with no noun after it ("move the  to the box"); case-sensitive so letters ("the letter A.") don't count
EMPTY_SLOT = re.compile(r"\b(?:[Tt]he|an?)\s+(?:to|with|on|onto|in|into|from|of|and|at|by|for)\b|\b[Tt]he\s*(?:[,.;:!?]|$)")
REPEATED_WORD = re.compile(r"\b([A-Za-z]{2,})\s+\1\b", re.I)  # "with with", "the the"
SLUG = re.compile(r"^[\w-]*[A-Za-z][-_][A-Za-z][\w-]*$")  # whole prompt is a file/task name
ROBOT_PREFIX = re.compile(r"^[A-Z][a-z]+_[A-Z]\d+_")  # "Galbot_G1_"
JUNK_SUFFIX = re.compile(r"(?:_(?:new\d*|\d+(?:_\d+)*)|\d*(?:Copy)+|\s+v\d+|·)$")  # "_new1", "_0501_04", "2Copy", " v3"
# two-letter asset variant code glued to a name: "microwave_gr"
ASSET_CODE = re.compile(r"\b([A-Za-z]{3,})_(?!(?:up|on|in|to|of|at|by|it|an|or|is|as|go|do)\b)[a-z]{2}\b")
UNDERSCORE_TOKEN = re.compile(r"\b[A-Za-z0-9]+_\w+")
PLEASE = re.compile(r"^please[,\s]+|,?\s*\bplease\b", re.I)


def has_asset_id(text: str) -> bool:
    """Robot prefix, recording/version suffix, variant code, or an identifier inside a sentence."""
    t = text.strip()
    return bool(
        ROBOT_PREFIX.match(t)
        or JUNK_SUFFIX.search(t)
        or ASSET_CODE.search(t)
        or (" " in t and UNDERSCORE_TOKEN.search(t))
    )


def normalize_prompt(text: str) -> str:
    """One house style: words not identifiers, no "please", single spaces,
    capitalized first letter, terminal punctuation."""
    t = ROBOT_PREFIX.sub("", " ".join(text.split()))
    while JUNK_SUFFIX.search(t):  # suffixes stack: "_2CopyCopy", "_0501_02·"
        t = JUNK_SUFFIX.sub("", t).rstrip("_ ")
    t = ASSET_CODE.sub(r"\1", t)
    if " " not in t:
        t = t.replace("-", " ")
    t = PLEASE.sub("", t.replace("_", " "))
    t = " ".join(t.split()).strip(" ,")
    if not t:
        return text.strip()
    t = t[0].upper() + t[1:]
    return t if re.search(r"[.!?。]$", t) else t + "."
