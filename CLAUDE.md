# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project that evaluates how different LLMs (GPT, Gemini, Claude, Mistral) respond to scientific prompts, specifically studying whether they cite peer-reviewed sources. The pipeline goes: **templates → prompts → LLM responses → URL/reference analysis**.

## Environment Setup

This project uses `uv` for dependency management with Python 3.12.

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv sync
```

API keys are loaded from a `.env` file via `python-dotenv`. Required keys: `GEMINI_API_KEY`, `OPENAI_API_KEY` (or `GPT_API_KEY`), and similar for other providers.

## Common Commands

```bash
# Generate prompts from the Excel templates file
python src/create_prompts/templates_to_prompts.py data/templates.xlsx data/prompts.csv

# Run an LLM on a prompts CSV (run from the src/run_llms/ directory)
cd src/run_llms
python run_llm.py --model_id gpt --prompts_path ../../data/prompts.csv --output_path ../../data/results/results_gpt.csv --save_every 10

# Classify URLs from a text file (one URL per line)
python src/process_results/classify_urls.py data/processed/urls1.txt --output results.csv --workers 8
```

Valid `--model_id` values: `gpt`, `gemini`, `claude`, `mistral`.

## Architecture

### Data Flow

1. **`data/templates.xlsx`** — Excel workbook with 3 sheets: `Templates` (prompt templates with `[TOPIC]`/`[LINK]`/`[TERM]` placeholders), `Statements` (scientific claims), `Terms`
2. **`src/create_prompts/templates_to_prompts.py`** — Expands templates × statements into a flat `prompts.csv`
3. **`src/run_llms/`** — Runs prompts against LLM APIs, saves results to `data/results/results_<model>.csv` (columns: `prompt`, `result`, `references`, `tokens`). Supports checkpointing: if the output CSV already exists, it resumes from incomplete rows.
4. **`scripts/*.ipynb`** — Jupyter notebooks for post-processing and analysis of raw results
5. **`src/process_results/`** — Library modules used by notebooks for URL validation and peer-review classification

### LLM Runner Pattern

`src/run_llms/runner.py` defines the abstract `LLMRunner` base class. Each model has its own subclass (`GeminiRunner`, `GPTRunner`, `ClaudeRunner`, `MistralRunner`) implementing `connect()` and `run_one_prompt()`. All runners return `(answer_text, urls_list, raw_response)` from `run_one_prompt()`.

The `run_llm.py` entrypoint must be run **from the `src/run_llms/` directory** because runners import each other with relative imports (e.g., `from runner import LLMRunner`).

### URL/Reference Processing

- **`exists_url.py`** — Validates URL reachability via GET requests (with thread-local sessions for parallel use). Status codes 402/403/405/406/502/503 are treated as "exists" (paywalls/server errors, not missing content).
- **`classify_urls.py`** — Classifies URLs as peer-reviewed using a layered strategy: (1) known domain allowlist, (2) DOI extraction → Crossref API lookup, (3) URL heuristics.
- **`source_identification.py`** — More detailed peer-review classification (refereed / not_refereed / unknown) with Crossref metadata; also contains `refine_unknowns_df()` to reclassify unknowns by fetching page HTML for embedded DOIs.
- **`utils.py`** — Fixes `exists=False` → `exists=True` for recoverable HTTP status codes in serialized cell values.
