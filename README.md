# GenAI-Science

This repository presents the analysis conducted to examine how large language models (GPT, Gemini, Mistral) behave when asked to provide references for scientific claims.

The experiment was designed to study how generative AI systems handle research-oriented questions in science communication contexts, recreating the kinds of situations a journalist or communicator might face when looking for sources to support a claim, explain a concept, or summarise a debate. 

## Pipeline

The analysis follows these stages:

```
templates.xlsx → prompts.csv → LLM responses → URL/reference analysis → results
```

1. Prompt templates with topic/statement placeholders are expanded into a flat CSV.
2. Each prompt is sent to one or more LLM APIs; responses and extracted URLs are saved.
3. URLs are validated (reachable?) and classified (peer-reviewed?).
4. Notebooks aggregate and visualize the results.

## Requirements

- Python 3.12–3.13
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- API keys for the models you intend to run

## Setup

```bash
# Clone the repo and enter the directory
git clone <repo-url>
cd GenAI-Science

# Create the virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Copy the example env file and fill in your API keys
cp .env.example .env
```

Required keys in `.env`:

| Variable | Provider |
|---|---|
| `OPENAI_API_KEY` | OpenAI (GPT) |
| `GEMINI_API_KEY` | Google Gemini |
| `MISTRAL_API_KEY` | Mistral AI |

## Usage

### 1. Generate prompts

Expand the Excel templates into a flat prompts CSV:

```bash
python src/create_prompts/templates_to_prompts.py data/templates.xlsx data/prompts.csv
```

### 2. Run an LLM

Must be run from the `src/run_llms/` directory (relative imports):

```bash
cd src/run_llms
python run_llm.py \
  --model_id gpt \
  --prompts_path ../../data/prompts.csv \
  --output_path ../../data/results/results_gpt.csv \
  --save_every 10
```

Valid `--model_id` values: `gpt`, `gemini`, `claude`, `mistral`.

If the output CSV already exists, the runner resumes from incomplete rows (checkpointing).

### 3. Classify URLs

```bash
python src/process_results/classify_urls.py data/processed/urls.txt \
  --output results.csv \
  --workers 8
```

### 4. Analysis

Open the notebooks in `scripts/` with Jupyter to run post-processing and generate visualizations:

```bash
jupyter notebook scripts/
```

## Project Structure

```
GenAI-Science/
├── data/
│   ├── templates.xlsx          # Prompt templates, statements, and terms
│   ├── prompts.csv             # Generated prompts (output of step 1)
│   ├── results/                # Raw LLM outputs (one CSV per model)
│   └── analisis/               # Processed/analyzed data
├── scripts/                    # Jupyter notebooks for analysis
├── src/
│   ├── create_prompts/
│   │   └── templates_to_prompts.py
│   ├── run_llms/
│   │   ├── runner.py           # Abstract LLMRunner base class
│   │   ├── run_llm.py          # CLI entrypoint
│   │   ├── openai_runner.py
│   │   ├── gemini_runner.py
│   │   ├── claude_runner.py
│   │   └── mistral_runner.py
│   └── process_results/
│       ├── exists_url.py       # URL reachability validation
│       ├── classify_urls.py    # Peer-review classification
│       ├── source_identification.py
│       └── utils.py
└── pyproject.toml
```

## Output Format

Each `data/results/results_<model>.csv` contains:

| Column | Description |
|---|---|
| `prompt` | The prompt sent to the model |
| `result` | The model's text response |
| `references` | List of URLs extracted from the response |
| `tokens` | Token usage for the request |

## URL Classification

URLs extracted from responses are classified using a layered strategy:

1. **Known-domain allowlist** — fast path for well-known academic publishers
2. **DOI extraction + Crossref API** — resolves DOIs to verify peer-review status
3. **URL heuristics** — pattern matching for journal/repository URLs

HTTP status codes 402, 403, 405, 406, 502, and 503 are treated as "exists" (paywalls or transient server errors, not missing content).
