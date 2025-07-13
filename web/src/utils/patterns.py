from utils.constants import NON_KETO, NON_VEGAN, KETO_WHITELIST, VEGAN_WHITELIST
import re
from typing import List

# ============================================================================
# PATTERN COMPILATION
# ============================================================================

def compile_any(words: List[str]) -> re.Pattern:
    """Compile a list of words into a single regex pattern."""
    return re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, words)), re.I)

# Compile patterns once
RX_KETO = compile_any(NON_KETO)
RX_WL_KETO = re.compile("|".join(KETO_WHITELIST), re.I)
RX_VEGAN = compile_any(NON_VEGAN)
RX_WL_VEGAN = re.compile("|".join(VEGAN_WHITELIST), re.I)
_PORTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?|\d+\s*/\s*\d+)\s*([a-zA-Z]+)\b", re.A)
