import re
import requests
from functools import lru_cache

DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"<>]+', re.I)

def extract_doi(text_or_url: str) -> str | None:
    if not text_or_url:
        return None
    m = DOI_RE.search(str(text_or_url))
    return m.group(0).rstrip(').,;:]') if m else None

@lru_cache(maxsize=50_000)
def crossref_metadata(doi: str) -> dict | None:
    r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=20, headers={"User-Agent": "link-classifier/1.0"})
    if r.status_code != 200:
        return None
    return r.json().get("message")

def classify_link_peer_review(url: str) -> dict:
    u = (url or "").lower()

    # Preprints: normalmente no peer-reviewed
    if any(d in u for d in ["arxiv.org", "ssrn.com", "biorxiv.org", "medrxiv.org"]):
        return {"label": "not_refereed", "reason": "preprint_server", "confidence": 0.95}

    doi = extract_doi(url)
    if not doi:
        return {"label": "unknown", "reason": "no_doi", "confidence": 0.3}

    meta = crossref_metadata(doi)
    if not meta:
        return {"label": "unknown", "reason": "doi_not_resolved", "confidence": 0.3}

    work_type = meta.get("type")  # e.g. journal-article, proceedings-article
    container = (meta.get("container-title") or [None])[0]
    publisher = meta.get("publisher")
    issn = meta.get("ISSN", []) or []
    isbn = meta.get("ISBN", []) or []

    # Tipos que vamos a considerar "refereed" (journal + proceedings)
    refereed_types = {"journal-article", "proceedings-article"}

    if work_type in refereed_types:
        # Confianza mayor para journal; un poco menor para proceedings por variabilidad entre conferencias
        conf = 0.9 if work_type == "journal-article" else 0.75
        return {
            "label": "refereed",
            "reason": f"crossref_type_{work_type}",
            "confidence": conf,
            "container": container,
            "publisher": publisher,
            "issn": issn,
            "isbn": isbn,
            "doi": doi,
        }

    # Otros tipos: book-chapter, posted-content, report, etc.
    return {
        "label": "unknown",
        "reason": f"crossref_type_{work_type}",
        "confidence": 0.4,
        "container": container,
        "publisher": publisher,
        "doi": doi,
    }


# df["url_labels"] = df["urls"].apply(lambda lst: [classify_link_peer_review(u) for u in lst])
# "urls" es el nombre de la columna donde están las listas de URLs extraídas de las respuestas del modelo. El resultado es una nueva columna "url_labels" con la clasificación de cada URL.

import pandas as pd

def extract_labels_per_row(url_label_dicts) -> list[str]:
    """
    Recibe una lista de dicts (uno por URL) y devuelve una lista de labels.
    Maneja None/NaN y entradas raras.
    """
    if url_label_dicts is None or (isinstance(url_label_dicts, float) and pd.isna(url_label_dicts)):
        return []
    if not isinstance(url_label_dicts, (list, tuple)):
        return []

    labels = []
    for item in url_label_dicts:
        if isinstance(item, dict) and "label" in item:
            labels.append(item["label"])
    return labels

def label_distribution(df: pd.DataFrame, col_url_labels: str, *, by="url") -> pd.DataFrame:
    """
    Calcula la distribución de labels.

    by="url": cuenta cada URL clasificada (aplana listas) => distribución sobre todas las URLs.
    by="row": cuenta 1 label por fila (mayoría; si empate, 'mixed' o 'unknown').

    Devuelve DataFrame con count y pct.
    """
    if by not in {"url", "row"}:
        raise ValueError("by debe ser 'url' o 'row'")

    if by == "url":
        # Aplanar: cada URL aporta 1 label
        all_labels = []
        for lst in df[col_url_labels].apply(extract_labels_per_row):
            all_labels.extend(lst)

        s = pd.Series(all_labels, name="label")
        counts = s.value_counts(dropna=False)

    else:
        # Un label por fila (resumen)
        def row_label(lst):
            labels = extract_labels_per_row(lst)
            if not labels:
                return "unknown"
            vc = pd.Series(labels).value_counts()
            if len(vc) == 1:
                return vc.index[0]
            # mayoría clara
            if vc.iloc[0] > vc.iloc[1]:
                return vc.index[0]
            return "mixed"

        s = df[col_url_labels].apply(row_label)
        counts = s.value_counts(dropna=False)

    dist = counts.rename("count").to_frame()
    dist["pct"] = (dist["count"] / dist["count"].sum()).round(4)
    dist.index.name = "label"
    return dist.reset_index()