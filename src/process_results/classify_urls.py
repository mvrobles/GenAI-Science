## FROM CLAUDE

"""
classify_urls.py
----------------
Clasifica una lista de URLs como fuentes académicas con revisión por pares (sí/no).

Estrategia en capas:
  1. Dominio conocido (lista blanca)
  2. Patrón DOI en la URL → consulta Crossref para confirmar peer-review
  3. Heurística de patrones en la URL

USO:
  python classify_urls.py urls.txt
  python classify_urls.py urls.txt --output resultados.csv
  python classify_urls.py urls.txt --workers 10   # paralelismo

El archivo de entrada debe tener una URL por línea.
"""

import argparse
import csv
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

# ─────────────────────────────────────────────
# CAPA 1: DOMINIOS CONOCIDOS
# ─────────────────────────────────────────────

# Dominios que SIEMPRE son peer-reviewed
PEER_REVIEWED_DOMAINS = {
    # Agregadores y bases de datos
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "europepmc.org",
    "jstor.org",
    "sciencedirect.com",
    "springer.com",
    "link.springer.com",
    "wiley.com",
    "onlinelibrary.wiley.com",
    "tandfonline.com",
    "sagepub.com",
    "journals.sagepub.com",
    "oup.com",
    "academic.oup.com",
    "cambridge.org",
    "journals.cambridge.org",
    "nature.com",
    "cell.com",
    "bmj.com",
    "thelancet.com",
    "nejm.org",
    "jamanetwork.com",
    "pnas.org",
    "science.org",
    "sciencemag.org",
    "plos.org",
    "journals.plos.org",
    "mdpi.com",
    "frontiersin.org",
    "hindawi.com",
    "biomed central.com",
    "biomedcentral.com",
    "karger.com",
    "elsevier.com",
    "acm.org",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "ascelibrary.org",
    "pubs.acs.org",
    "rsc.org",
    "embopress.org",
    "rupress.org",
    "genetics.org",
    "bloodjournal.org",
    "ahajournals.org",
    "diabetesjournals.org",
    "annualreviews.org",
    "royalsocietypublishing.org",
    "physiology.org",
    "fasebj.org",
    "jneurosci.org",
    "asm.org",
    "microbiology.org",
    "biochemj.org",
    "portlandpress.com",
    "geoscienceworld.org",
    "agu.org",
    "ametsoc.org",
    "ams.org",
    "siam.org",
}

# Dominios académicos que pueden o NO ser peer-reviewed
# (repositorios de preprints, redes académicas, etc.)
ACADEMIC_NOT_PEER_REVIEWED = {
    "arxiv.org",
    "ssrn.com",
    "osf.io",
    "preprints.org",
    "biorxiv.org",
    "medrxiv.org",
    "chemrxiv.org",
    "psyarxiv.com",
    "eartharxiv.org",
    "techrxiv.org",
    "researchsquare.com",
    "authorea.com",
    "zenodo.org",          # puede tener ambos tipos
    "figshare.com",
    "researchgate.net",    # red social académica, no revisa pares
    "academia.edu",
    "semanticscholar.org", # indexador, no editorial
    "scholar.google.com",
    "base-search.net",
    "core.ac.uk",
    "openalex.org",
    "unpaywall.org",
    "scihub",
}

# ─────────────────────────────────────────────
# CAPA 2: EXTRACCIÓN DE DOI
# ─────────────────────────────────────────────

DOI_PATTERNS = [
    re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE),
    re.compile(r"doi\.org/(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE),
]

def extract_doi(url: str) -> str | None:
    """Extrae un DOI de la URL si existe."""
    for pat in DOI_PATTERNS:
        m = pat.search(url)
        if m:
            doi = m.group(0)
            # Normalizar: quitar prefijo doi.org/
            doi = re.sub(r"^.*doi\.org/", "", doi, flags=re.IGNORECASE)
            # Limpiar caracteres finales
            doi = doi.rstrip(".,;)")
            return doi
    return None

# ─────────────────────────────────────────────
# CAPA 3: CROSSREF API
# ─────────────────────────────────────────────

CROSSREF_BASE = "https://api.crossref.org/works/"
CROSSREF_HEADERS = {"User-Agent": "URLClassifier/1.0 (mailto:tu@email.com)"}

def check_doi_crossref(doi: str) -> tuple[bool, str]:
    """
    Consulta Crossref para verificar si el DOI corresponde a un artículo
    con revisión por pares.
    Retorna (es_peer_reviewed, razon).
    """
    try:
        url = CROSSREF_BASE + requests.utils.quote(doi, safe="")
        r = requests.get(url, headers=CROSSREF_HEADERS, timeout=8)
        if r.status_code == 404:
            return False, "DOI no encontrado en Crossref"
        if r.status_code != 200:
            return False, f"Crossref error {r.status_code}"

        data = r.json().get("message", {})
        doc_type = data.get("type", "").lower()

        # Tipos que implican revisión por pares
        PEER_REVIEWED_TYPES = {
            "journal-article",
            "proceedings-article",
            "book-chapter",
            "monograph",
            "edited-book",
            "reference-entry",
        }
        # Tipos que NO tienen revisión por pares
        NON_PEER_REVIEWED_TYPES = {
            "posted-content",   # preprints
            "dataset",
            "component",
            "report",
            "report-series",
            "standard",
        }

        if doc_type in PEER_REVIEWED_TYPES:
            journal = data.get("container-title", [""])[0] if data.get("container-title") else ""
            return True, f"Crossref: tipo '{doc_type}'" + (f", revista: {journal}" if journal else "")
        elif doc_type in NON_PEER_REVIEWED_TYPES:
            return False, f"Crossref: tipo '{doc_type}' (no peer-reviewed)"
        else:
            return False, f"Crossref: tipo desconocido '{doc_type}'"

    except requests.exceptions.Timeout:
        return False, "Crossref: timeout"
    except Exception as e:
        return False, f"Crossref error: {str(e)[:50]}"

# ─────────────────────────────────────────────
# CAPA 4: HEURÍSTICA DE URL
# ─────────────────────────────────────────────

ACADEMIC_URL_SIGNALS = [
    re.compile(r"/article/", re.I),
    re.compile(r"/abstract/", re.I),
    re.compile(r"/fulltext/", re.I),
    re.compile(r"/journal[s]?/", re.I),
    re.compile(r"/paper[s]?/", re.I),
    re.compile(r"/publication[s]?/", re.I),
    re.compile(r"/proceedings/", re.I),
    re.compile(r"/pmc/articles/", re.I),
    re.compile(r"pmc\d+", re.I),
]

def heuristic_check(url: str, parsed) -> tuple[bool, str]:
    """Señales heurísticas en la URL."""
    path = parsed.path.lower()

    # Dominios .edu o .ac.XX con ruta de artículo
    tld_parts = parsed.netloc.split(".")
    if any(p in ("edu", "ac") for p in tld_parts):
        for sig in ACADEMIC_URL_SIGNALS:
            if sig.search(url):
                return True, "Heurística: dominio edu/ac + patrón de artículo"
        return False, "Heurística: dominio edu/ac sin patrón de artículo"

    for sig in ACADEMIC_URL_SIGNALS:
        if sig.search(url):
            return True, f"Heurística: patrón '{sig.pattern}' en URL"

    return False, "Sin señales académicas detectadas"

# ─────────────────────────────────────────────
# CLASIFICADOR PRINCIPAL
# ─────────────────────────────────────────────

def classify_url(url: str) -> dict:
    url = url.strip()
    result = {
        "url": url,
        "peer_reviewed": "no",
        "confianza": "baja",
        "razon": "",
        "doi": "",
    }

    if not url or not url.startswith("http"):
        result["razon"] = "URL inválida o vacía"
        return result

    try:
        parsed = urlparse(url)
    except Exception:
        result["razon"] = "Error al parsear URL"
        return result

    netloc = parsed.netloc.lower().lstrip("www.")

    # ── Capa 1a: dominio claramente peer-reviewed ──
    for domain in PEER_REVIEWED_DOMAINS:
        if netloc == domain or netloc.endswith("." + domain):
            result["peer_reviewed"] = "sí"
            result["confianza"] = "alta"
            result["razon"] = f"Dominio académico conocido: {domain}"
            return result

    # ── Capa 1b: dominio académico SIN peer-review ──
    for domain in ACADEMIC_NOT_PEER_REVIEWED:
        if netloc == domain or netloc.endswith("." + domain):
            # Excepción: si tiene DOI, verificar con Crossref
            doi = extract_doi(url)
            if doi:
                result["doi"] = doi
                is_pr, razon = check_doi_crossref(doi)
                result["peer_reviewed"] = "sí" if is_pr else "no"
                result["confianza"] = "alta" if is_pr else "media"
                result["razon"] = razon
            else:
                result["peer_reviewed"] = "no"
                result["confianza"] = "alta"
                result["razon"] = f"Repositorio/red académica sin peer-review: {domain}"
            return result

    # ── Capa 2: DOI en la URL → Crossref ──
    doi = extract_doi(url)
    if doi:
        result["doi"] = doi
        is_pr, razon = check_doi_crossref(doi)
        result["peer_reviewed"] = "sí" if is_pr else "no"
        result["confianza"] = "alta" if is_pr else "media"
        result["razon"] = razon
        return result

    # ── Capa 4: Heurística ──
    is_pr, razon = heuristic_check(url, parsed)
    result["peer_reviewed"] = "sí" if is_pr else "no"
    result["confianza"] = "baja"
    result["razon"] = razon
    return result

# ─────────────────────────────────────────────
# PROCESAMIENTO EN LOTE
# ─────────────────────────────────────────────

def process_urls(urls: list[str], workers: int = 8) -> list[dict]:
    results = []
    total = len(urls)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(classify_url, url): url for url in urls}
        done = 0
        for future in as_completed(futures):
            done += 1
            res = future.result()
            results.append(res)
            # Progreso en consola
            bar = int(30 * done / total)
            print(f"\r[{'█'*bar}{'░'*(30-bar)}] {done}/{total}  ", end="", flush=True)

    print()  # newline al terminar
    # Ordenar por URL original para mantener orden
    url_order = {url: i for i, url in enumerate(urls)}
    results.sort(key=lambda r: url_order.get(r["url"], 9999))
    return results

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clasifica URLs como fuentes académicas con revisión por pares."
    )
    parser.add_argument("input", help="Archivo .txt con una URL por línea")
    parser.add_argument("--output", default="resultados_clasificacion.csv",
                        help="Archivo CSV de salida (default: resultados_clasificacion.csv)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Número de hilos paralelos (default: 8)")
    args = parser.parse_args()

    # Leer URLs
    try:
        with open(args.input, encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {args.input}")
        sys.exit(1)

    print(f"📋 {len(urls)} URLs cargadas. Clasificando con {args.workers} hilos...\n")
    start = time.time()

    results = process_urls(urls, workers=args.workers)

    elapsed = time.time() - start
    print(f"⏱  Tiempo total: {elapsed:.1f}s\n")

    # Estadísticas
    si = sum(1 for r in results if r["peer_reviewed"] == "sí")
    no = len(results) - si
    print(f"✅ Peer-reviewed:     {si:>5} ({100*si/len(results):.1f}%)")
    print(f"❌ No peer-reviewed:  {no:>5} ({100*no/len(results):.1f}%)")

    # Guardar CSV
    fieldnames = ["url", "peer_reviewed", "confianza", "razon", "doi"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n💾 Resultados guardados en: {args.output}")

if __name__ == "__main__":
    main()