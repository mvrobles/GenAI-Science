import ast

RECOVERABLE_CODES = {402, 403, 406, 502, 503, 405}

def fix_exists_by_status(cell_value: str) -> str:
    """
    Recibe el contenido de una celda (string con lista de dicts),
    cambia exists=False a exists=True para status codes recuperables,
    y devuelve el string actualizado.
    """
    try:
        records = ast.literal_eval(cell_value)
    except (ValueError, SyntaxError):
        return cell_value  # si no se puede parsear, devuelve tal cual

    for record in records:
        if (
            not record.get("exists", True)
            and record.get("status_code") in RECOVERABLE_CODES
        ):
            record["exists"] = True

    return str(records)

