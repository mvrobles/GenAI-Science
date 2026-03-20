import pandas as pd
import requests
import ast
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import threading

# 1) Session "thread-local" (una por hilo)
_thread_local = threading.local()

def to_list_safe(x):
    """
    Convierte:
    - NaN/None -> []
    - lista -> lista
    - string tipo "['a','b']" -> lista
    - string 'nan' / '' -> []
    - cualquier otro -> []
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() == "nan":
            return []
        # si parece lista serializada
        if s.startswith("[") and s.endswith("]"):
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return []
        return []
    return []


def url_exists(url, timeout: int = 15, allow_redirects: bool = True,
               accept_3xx: bool = False, max_bytes: int = 2048) -> dict:
    """
    Valida acceso real a la URL completa (GET, no solo HEAD).
    """
    # Normaliza / filtra NaN y valores raros
    if url is None:
        return {"url": url, "exists": False, "status_code": None, "final_url": None, "error": "empty"}
    if isinstance(url, float) and pd.isna(url):
        return {"url": None, "exists": False, "status_code": None, "final_url": None, "error": "empty"}
    if not isinstance(url, str):
        url = str(url)

    url = url.strip()
    if url == "" or url.lower() == "nan":
        return {"url": url, "exists": False, "status_code": None, "final_url": None, "error": "empty"}

    try:
        with requests.get(
            url,
            timeout=timeout,
            allow_redirects=allow_redirects,
            headers={"User-Agent": "url-validator/1.0"},
            stream=True,
        ) as r:
            status = r.status_code

            # Lee un poquito para confirmar acceso sin bajar todo
            try:
                for chunk in r.iter_content(chunk_size=max_bytes):
                    if chunk:
                        break
            except Exception:
                pass

            if 200 <= status < 300:
                exists = True
            elif accept_3xx and 300 <= status < 400:
                exists = True
            elif status in [402, 403, 406, 502, 503, 405]: 
                exists = True
            elif status is None:
                exists = None
            else:
                exists = False

            return {
                "url": url,
                "exists": exists,
                "status_code": status,
                "final_url": str(r.url),
                "error": None,
            }

    except requests.exceptions.RequestException as e:
        return {
            "url": url,
            "exists": False,
            "status_code": None,
            "final_url": None,
            "error": f"{type(e).__name__}: {e}",
        }

def url_exists_list(urls, **kwargs) -> list[dict]:
    """
    Aplica url_exists a una lista, tolerante a NaN y strings serializados.
    """
    urls = to_list_safe(urls)  # <- usa la función de arriba
    return [url_exists(u, **kwargs) for u in urls if u not in (None, "")]




def get_session():
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        # opcional: headers/UA
        s.headers.update({"User-Agent": "Mozilla/5.0"})
        _thread_local.session = s
    return _thread_local.session

def limpiar_url(url: str) -> str:
    return url.replace('"', "").replace("'", "").strip()

def obtener_url_real(url: str) -> str:
    try:
        s = get_session()
        r = s.head(url, allow_redirects=True, timeout=5)
        return r.url
    except requests.RequestException as e:
        return f"Error: {e.__class__.__name__}: {e}"

# 2) paraleliza *dentro* de cada lista
def obtener_urls(list_url, max_workers=32):
    urls = [limpiar_url(u) for u in list_url]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(obtener_url_real, urls))

# 3) apply con progress a nivel fila
def apply_obtener_urls_con_progress(df, col="references", max_workers_urls=32):
    out = []
    for lst in tqdm(df[col].tolist(), total=len(df), desc="Filas"):
        out.append(obtener_urls(lst, max_workers=max_workers_urls))
    return out
