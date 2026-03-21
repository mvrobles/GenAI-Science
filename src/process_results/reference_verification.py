import re
import difflib
import requests
import pandas as pd
from functools import lru_cache
from tqdm import tqdm
# ---------------------------------------------------------------------------
# Step 1: Extract references from result text
# ---------------------------------------------------------------------------

# Matches: "1. ...", "[1] ...", or "1 "Title..." patterns
_REF_SPLIT_RE = re.compile(r'(?:^|\n)\s*(?:\[?\d+\]?\.?\s+)', re.MULTILINE)
# Trailing citation markers like [1], [2, 3], etc.
_TRAILING_MARKER_RE = re.compile(r'\s*\[\d+(?:,\s*\d+)*\]\s*$')


def extract_references(text: str) -> list[str]:
    """
    Extract numbered bibliographic references from Mistral result text.
    Handles formats: "1. Citation [1]", "[1] Citation", "1 "Title" ..."
    Returns list of cleaned citation strings.
    """
    if not text or (isinstance(text, float) and pd.isna(text)):
        return []

    parts = _REF_SPLIT_RE.split(str(text))
    refs = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Remove trailing [N] markers
        part = _TRAILING_MARKER_RE.sub('', part).strip()
        # Skip very short strings (not real citations)
        if len(part) < 20:
            continue
        # Skip if it looks like prose (no author/title signal)
        # Keep if it has a year, "by", quoted text, or journal-like pattern
        if any(sig in part for sig in ['"', 'by ', 'vol.', 'Vol.', 'pp.', 'doi:', 'DOI:', '19', '20']):
            refs.append(part)

    return refs


# ---------------------------------------------------------------------------
# Step 2: Extract title from citation string
# ---------------------------------------------------------------------------

_QUOTED_TITLE_RE = re.compile(r'"([^"]{10,})"')
# APA format: "Author, A., & Author, B. (YEAR). Title. Venue, ..."
_APA_YEAR_RE = re.compile(r'\(\d{4}\)\.\s+')
# Next sentence boundary: ". Capital letter" (signals start of venue)
_DOT_CAPITAL_RE = re.compile(r'\.\s+(?=[A-Z])')


def _apa_parts(citation: str) -> tuple[str, str]:
    """
    If citation is APA format (contains '(YYYY). '), return (title, venue).
    Otherwise return ("", "").
    """
    m = _APA_YEAR_RE.search(citation)
    if not m:
        return "", ""

    rest = citation[m.end():]  # everything after "(YEAR). "

    # Title ends at first ". Capital" boundary
    boundary = _DOT_CAPITAL_RE.search(rest)
    if boundary:
        title = rest[:boundary.start()].strip()
        after_title = rest[boundary.end():]  # venue onwards
        # Venue ends at first comma+digit, "vol.", "no.", or end of string
        venue_end = re.search(r',\s*(?:\d|\()|vol\.|no\.|pp\.|p\.', after_title, re.IGNORECASE)
        venue = after_title[:venue_end.start()].strip() if venue_end else after_title.strip()
    else:
        title = rest.strip()
        venue = ""

    return title, venue


def extract_title(citation: str) -> str:
    """
    Extract title from a bibliographic citation string.
    - APA format (Author (Year). Title. Venue) → extract from after year marker
    - Quoted text → return first quoted string
    - Else → text up to '. by' or first '. '
    """
    if not citation:
        return ""

    # APA: Author (Year). Title. Venue.
    title, _ = _apa_parts(citation)
    if title:
        return title

    # Quoted title: "Title" by Author...
    m = _QUOTED_TITLE_RE.search(citation)
    if m:
        return m.group(1).strip()

    # Fallback: up to ". by" or first ". "
    dot_by = citation.find('. by ')
    if dot_by != -1:
        return citation[:dot_by].strip()

    dot = citation.find('. ')
    if dot != -1:
        return citation[:dot].strip()

    return citation.strip()


def extract_venue(citation: str) -> str:
    """
    Extract the journal/venue name from a bibliographic citation string.
    Returns empty string if not found.
    """
    if not citation:
        return ""

    # APA: Author (Year). Title. Venue, vol(issue), pages.
    _, venue = _apa_parts(citation)
    if venue:
        return venue

    # "Title" by Author. Journal, vol. X
    m = re.search(
        r'\.\s+([A-Z][^,\.]{3,60}?)(?:,\s*(?:vol\.|Vol\.|v\.|no\.|pp\.|p\.|[0-9]{4}))',
        citation, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # "Title". Journal, year
    m = re.search(r'["\)\.]\s+([A-Z][^,\.]{3,60}?),\s*\d', citation)
    if m:
        candidate = m.group(1).strip()
        if len(candidate) > 5:
            return candidate

    return ""


def venue_similarity(citation: str, crossref_result: dict) -> float:
    """
    Compare extracted venue from citation vs CrossRef container-title.
    Returns ratio in [0, 1].
    """
    venue_a = extract_venue(citation).lower()
    venue_b = (crossref_result.get("container") or "").lower()
    if not venue_a or not venue_b:
        return 0.0
    return difflib.SequenceMatcher(None, venue_a, venue_b).ratio()


# ---------------------------------------------------------------------------
# Step 3: Search CrossRef by title
# ---------------------------------------------------------------------------

@lru_cache(maxsize=50_000)
def search_crossref(title: str, rows: int = 5) -> tuple[dict, ...]:
    """
    Search CrossRef by title. Returns tuple of result dicts (hashable for lru_cache).
    Each dict has keys: title, author, published_year, type, doi.
    """
    if not title:
        return ()
    try:
        r = requests.get(
            "https://api.crossref.org/works",
            params={"query.title": title, "rows": rows},
            timeout=20,
            headers={"User-Agent": "reference-verifier/1.0 (research project)"},
        )
        if r.status_code != 200:
            return ()
        items = r.json().get("message", {}).get("items", [])
        results = []
        for item in items:
            # Extract title (CrossRef returns a list)
            raw_title = item.get("title") or []
            item_title = raw_title[0] if raw_title else ""
            # Extract authors
            authors = item.get("author") or []
            author_str = "; ".join(
                f"{a.get('family', '')}".strip() for a in authors[:3]
            )
            # Extract year
            pub_date = item.get("published-print") or item.get("published-online") or {}
            date_parts = pub_date.get("date-parts") or [[]]
            year = date_parts[0][0] if date_parts and date_parts[0] else None
            container = item.get("container-title") or []
            results.append({
                "title": item_title,
                "author": author_str,
                "published_year": year,
                "type": item.get("type"),
                "doi": item.get("DOI"),
                "container": container[0] if container else "",
            })
        return tuple(results)
    except Exception:
        return ()


# ---------------------------------------------------------------------------
# Step 4: Compute similarity score
# ---------------------------------------------------------------------------

def citation_similarity(original_citation: str, crossref_result: dict) -> float:
    """
    Compare extracted title from citation vs CrossRef result title.
    Returns ratio in [0, 1] using SequenceMatcher.
    """
    title_a = extract_title(original_citation).lower()
    title_b = (crossref_result.get("title") or "").lower()
    if not title_a or not title_b:
        return 0.0
    return difflib.SequenceMatcher(None, title_a, title_b).ratio()


# ---------------------------------------------------------------------------
# Step 5: Classify each reference
# ---------------------------------------------------------------------------

def verify_reference(citation: str) -> dict:
    """
    Verify a single bibliographic citation against CrossRef.

    Returns dict with:
      citation, title_extracted, venue_extracted,
      top_match, title_similarity, venue_similarity, venue_crossref,
      status
    Status: "exists" | "review" | "hallucinated"
    """
    title = extract_title(citation)
    venue = extract_venue(citation)
    results = search_crossref(title)

    if not results:
        return {
            "citation": citation,
            "title_extracted": title,
            "venue_extracted": venue,
            "top_match": None,
            "title_similarity": 0.0,
            "venue_similarity": 0.0,
            "venue_crossref": "",
            "status": "hallucinated",
        }

    # Score all results by title similarity, pick best
    scored = [(citation_similarity(citation, r), r) for r in results]
    best_sim, best_match = max(scored, key=lambda x: x[0])

    vsim = venue_similarity(citation, best_match)
    venue_crossref = best_match.get("container") or ""

    if best_sim >= 0.85:
        status = "exists"
    elif best_sim >= 0.5:
        status = "review"
    else:
        status = "hallucinated"

    return {
        "citation": citation,
        "title_extracted": title,
        "venue_extracted": venue,
        "top_match": best_match,
        "title_similarity": round(best_sim, 4),
        "venue_similarity": round(vsim, 4),
        "venue_crossref": venue_crossref,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Step 6: Batch processing with checkpointing
# ---------------------------------------------------------------------------

def verify_references_df(
    df: pd.DataFrame,
    save_path: str,
    save_every: int = 50,
) -> pd.DataFrame:
    """
    For each row in df, explode references and run verify_reference() per citation.
    Saves progress CSV every save_every rows.

    Expects df to have a 'references_extracted' column (list of citation strings).
    Returns long-form DataFrame with one row per citation.
    """
    # Build list of (original_index, prompt, citation) tuples
    rows = []
    for idx, row in df.iterrows():
        refs = row.get("references_extracted") or []
        for ref in refs:
            rows.append({"original_index": idx, "prompt": row.get("prompt", ""), "citation": ref})

    if not rows:
        return pd.DataFrame()

    results = []
    for i, entry in tqdm(enumerate(rows)):
        result = verify_reference(entry["citation"])
        results.append({
            "original_index": entry["original_index"],
            "prompt": entry["prompt"],
            **result,
        })

        if save_path and (i + 1) % save_every == 0:
            pd.DataFrame(results).to_csv(save_path, index=False)
            print(f"  Checkpoint saved at row {i + 1}/{len(rows)}")

    out_df = pd.DataFrame(results)
    if save_path:
        out_df.to_csv(save_path, index=False)
        print(f"Saved {len(out_df)} rows to {save_path}")

    return out_df
