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



### -------------- Depuración que no sé bien cómo usar o entender ------------------
import re
import pandas as pd
import requests
from urllib.parse import urlsplit
from functools import lru_cache

DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"<>]+', re.I)
URL_RE = re.compile(r'https?://[^\s<>"\']+')

def get_domain(url: str) -> str:
    try:
        return urlsplit(url).netloc.lower()
    except Exception:
        return ""

def safe_list(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    return x if isinstance(x, list) else []

def extract_urls_from_text(text) -> list[str]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    s = str(text)
    urls = []
    for m in URL_RE.finditer(s):
        u = m.group(0).strip()
        while u and u[-1] in ')]}.,;:':
            u = u[:-1]
        urls.append(u)
    # dedup preservando orden
    seen, out = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u); out.append(u)
    return out

def extract_doi(text_or_url: str) -> str | None:
    if not text_or_url:
        return None
    m = DOI_RE.search(str(text_or_url))
    return m.group(0).rstrip(').,;:]') if m else None


# -------------------------
# Fase 1: reglas rápidas
# -------------------------

# Ajusta estas listas a tu corpus real
NOT_REFEREED_DOMAINS = {
    "axios.com", "itpro.com", "medium.com", "substack.com",
    "blog.google", "openai.com", "learn.microsoft.com",
}

# patrones de paths típicos no-académicos
NOT_REFEREED_PATH_HINTS = (
    "/blog", "/news", "/press", "/press-releases", "/newsroom",
    "/training", "/docs", "/documentation", "/help", "/support"
)

PREPRINT_DOMAINS = {"arxiv.org", "ssrn.com", "biorxiv.org", "medrxiv.org"}

def fast_rule_label(url: str) -> dict | None:
    """
    Devuelve dict con label si detecta con alta confianza.
    Si no puede concluir, devuelve None.
    """
    u = (url or "").lower()
    dom = get_domain(u)
    path = urlsplit(u).path.lower()

    if dom in PREPRINT_DOMAINS:
        return {"label": "not_refereed", "reason": "preprint_server"}

    if dom in NOT_REFEREED_DOMAINS:
        return {"label": "not_refereed", "reason": "domain_rule"}

    if any(h in path for h in NOT_REFEREED_PATH_HINTS):
        return {"label": "not_refereed", "reason": "path_hint_rule"}

    return None


# -------------------------
# Fase 2: sacar DOI del HTML
# -------------------------

META_DOI_PATTERNS = [
    # citation_doi
    re.compile(r'<meta[^>]+name=["\']citation_doi["\'][^>]+content=["\']([^"\']+)["\']', re.I),
    # dc.identifier / dc.identifier.doi
    re.compile(r'<meta[^>]+name=["\']dc\.identifier(?:\.doi)?["\'][^>]+content=["\']([^"\']+)["\']', re.I),
    # og:doi (menos común)
    re.compile(r'<meta[^>]+property=["\']og:doi["\'][^>]+content=["\']([^"\']+)["\']', re.I),
]

@lru_cache(maxsize=50_000)
def fetch_html(url: str) -> str | None:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "unknown-refiner/1.0"})
        if r.status_code != 200:
            return None
        # Evita bajar PDFs enormes como HTML
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "application/pdf" in ctype:
            return None
        return r.text
    except Exception:
        return None

def doi_from_html(url: str) -> str | None:
    html = fetch_html(url)
    if not html:
        return None

    # 1) metas
    for pat in META_DOI_PATTERNS:
        m = pat.search(html)
        if m:
            candidate = m.group(1).strip()
            d = extract_doi(candidate)
            if d:
                return d

    # 2) regex general sobre el HTML (más ruidoso pero útil)
    d = extract_doi(html)
    return d


# -------------------------
# Crossref (para DOI -> tipo)
# -------------------------

@lru_cache(maxsize=50_000)
def crossref_metadata(doi: str) -> dict | None:
    try:
        r = requests.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=20,
            headers={"User-Agent": "link-classifier/1.0"}
        )
        if r.status_code != 200:
            return None
        return r.json().get("message")
    except Exception:
        return None

def classify_by_doi(doi: str) -> dict:
    meta = crossref_metadata(doi)
    if not meta:
        return {"label": "unknown", "reason": "doi_not_resolved", "doi": doi}

    work_type = meta.get("type")  # journal-article, proceedings-article, posted-content, etc.
    container = (meta.get("container-title") or [None])[0]
    publisher = meta.get("publisher")
    issn = meta.get("ISSN", []) or []
    isbn = meta.get("ISBN", []) or []

    if work_type == "journal-article":
        return {
            "label": "refereed",
            "reason": "crossref_journal_article",
            "doi": doi,
            "container": container,
            "publisher": publisher,
            "issn": issn,
            "isbn": isbn,
        }
    if work_type == "proceedings-article":
        return {
            "label": "refereed",
            "reason": "crossref_proceedings_article",
            "doi": doi,
            "container": container,
            "publisher": publisher,
            "issn": issn,
            "isbn": isbn,
        }

    return {
        "label": "unknown",
        "reason": f"crossref_type_{work_type}",
        "doi": doi,
        "container": container,
        "publisher": publisher,
        "issn": issn,
        "isbn": isbn,
    }


# -------------------------
# Diagnóstico de unknown
# -------------------------

def unknown_breakdown(df: pd.DataFrame, col_url_labels="url_labels") -> dict:
    """
    Devuelve:
      - dist por reason (solo unknown)
      - top dominios dentro de unknown
      - tasa unknown global (por URL)
    """
    rows = []
    for lst in df[col_url_labels].apply(safe_list):
        for item in lst:
            if isinstance(item, dict):
                url = item.get("url") or item.get("link") or ""
                label = item.get("label")
                reason = item.get("reason", "no_reason")
                rows.append({"url": url, "domain": get_domain(url), "label": label, "reason": reason})

    long = pd.DataFrame(rows)
    if long.empty:
        return {"unknown_reason_dist": pd.DataFrame(), "unknown_top_domains": pd.DataFrame(), "unknown_rate": 0.0}

    # unknown rate por URL
    unknown_rate = (long["label"].eq("unknown").mean())

    unk = long[long["label"] == "unknown"].copy()
    reason_dist = unk["reason"].value_counts().rename("count").to_frame()
    reason_dist["pct"] = (reason_dist["count"] / reason_dist["count"].sum()).round(4)
    reason_dist = reason_dist.reset_index().rename(columns={"index": "reason"})

    top_domains = unk["domain"].value_counts().head(25).rename("count").to_frame().reset_index().rename(columns={"index": "domain"})
    top_domains["pct"] = (top_domains["count"] / unk.shape[0]).round(4)

    return {
        "unknown_reason_dist": reason_dist,
        "unknown_top_domains": top_domains,
        "unknown_rate": float(round(unknown_rate, 4)),
    }


# -------------------------
# Refinamiento: actualiza unknowns
# -------------------------

def refine_unknown_item(item: dict) -> dict:
    """
    Toma un dict por URL y si está unknown, intenta reclasificar.
    Conserva la url.
    """
    if not isinstance(item, dict):
        return item

    label = item.get("label")
    url = item.get("url") or item.get("link") or item.get("source_url")
    if not url:
        return item

    # Solo trabajamos unknown
    if label != "unknown":
        # Asegura url persistente
        item["url"] = url
        return item

    # Fase 1: reglas rápidas
    fr = fast_rule_label(url)
    if fr is not None:
        out = {**item, **fr}
        out["url"] = url
        return out

    # Fase 2: DOI desde HTML (si no había DOI)
    doi = item.get("doi") or extract_doi(url)
    if not doi:
        doi = doi_from_html(url)
        if doi:
            item["doi"] = doi
            item["reason"] = "doi_found_in_html"

    if doi:
        classified = classify_by_doi(doi)
        out = {**item, **classified}
        out["url"] = url
        return out

    # Si nada funcionó, manten unknown pero actualiza razón
    out = dict(item)
    out["url"] = url
    out["reason"] = out.get("reason") or "unknown_no_signal"
    return out

def refine_unknowns_df(df: pd.DataFrame, col_url_labels="url_labels", new_col=None) -> pd.DataFrame:
    """
    Refina unknowns dentro de df[col_url_labels].
    Si new_col es None, sobrescribe col_url_labels.
    """
    target_col = new_col or col_url_labels

    def refine_list(lst):
        lst = safe_list(lst)
        return [refine_unknown_item(item) for item in lst]

    df[target_col] = df[col_url_labels].apply(refine_list)
    return df


# -------------------------
# Ejemplo de uso
# -------------------------
# 1) Diagnóstico inicial
# stats = unknown_breakdown(df, col_url_labels="url_labels")
# stats["unknown_rate"], stats["unknown_reason_dist"].head(), stats["unknown_top_domains"].head()

# 2) Refinar unknowns
# df = refine_unknowns_df(df, col_url_labels="url_labels")  # sobrescribe
# stats2 = unknown_breakdown(df, col_url_labels="url_labels")