# utils/query_flags.py
import re
from typing import List, Tuple

DietFlag = Tuple[str, float]      # e.g. ('keto', 1.0)  or  ('vegan', 0.75) or ('both', 0.8)

def split_query_flags(raw: str) -> Tuple[str, List[DietFlag]]:
    """
    Extract every   #keto[:threshold] / #vegan[:threshold] / #both[:threshold]   flag.

    Returns
    -------
    free_text : str            # the query with all flags stripped out
    flags     : list[DietFlag] # list of (diet, threshold) tuples
    
    Examples
    --------
    >>> split_query_flags("chicken #keto")
    ('chicken', [('keto', 1.0)])
    
    >>> split_query_flags("salad #both:0.8")
    ('salad', [('both', 0.8)])
    
    >>> split_query_flags("#vegan:0.9 #keto:0.7 pasta")
    ('pasta', [('vegan', 0.9), ('keto', 0.7)])
    """
    pattern = r'#(keto|vegan|both)(?::(\d*\.?\d+))?'
    flags: List[DietFlag] = []

    def _parse(m):
        diet = m.group(1).lower()
        thr  = float(m.group(2)) if m.group(2) else 1.0
        thr  = max(0.0, min(1.0, thr))        # clamp between 0 and 1
        flags.append((diet, thr))
        return ''                             # remove from text

    free_text = re.sub(pattern, _parse, raw, flags=re.I).strip()
    return free_text, flags