import re
from typing import Optional

TAXONOMY_PATTERN = re.compile(r"\b(?:kind|sig|area)/[a-zA-Z0-9_-]+\b", re.IGNORECASE)

def prepare_text(title: str, body: Optional[str]) -> str:
    """Prepare model input by combining, cleaning, and normalizing issue title and body.
    
    This matches the exact preprocessing contract used during model training:
    1. Removes command leakage (lines starting with '/').
    2. Removes taxonomy tokens (e.g., kind/*, sig/*, area/*).
    3. Normalizes whitespace (collapses spaces and newlines).
    4. Formats as 'Title: ...\\nBody: ...'.
    5. Truncates to a maximum of 2000 characters.
    """
    title_lines = [line.strip() for line in title.splitlines() if line.strip() and not line.strip().startswith("/")]
    clean_title = " ".join(title_lines)
    
    clean_body = ""
    if body is not None:
        body_lines = []
        for line in body.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("/"):
                body_lines.append(stripped)
        clean_body = "\n".join(body_lines)
        
    combined = f"Title: {clean_title}\nBody: {clean_body}"
    combined = TAXONOMY_PATTERN.sub("", combined)
    
    combined = re.sub(r"[ \t]+", " ", combined)
    combined = re.sub(r"\n+", "\n", combined)
    
    combined = combined.strip()
    return combined[:2000]
