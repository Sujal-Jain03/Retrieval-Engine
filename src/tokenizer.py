"""Shared tokenizer: lowercase, split on non-alphanumeric, drop empties.

No stemming, no stopwords — BM25 and dense both use these tokens.
(Named tokenizer.py, not tokenize.py: the stdlib has a module called
tokenize and the name collision broke imports.)"""

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
