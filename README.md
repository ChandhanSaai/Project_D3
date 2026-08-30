# D3: Debate, Deliberate, Decide

D3 orchestrates role-specialized LLM agents -- advocates, a judge, and an optional jury -- to evaluate LLM outputs through structured adversarial debate, inspired by courtroom proceedings.

## How It Works

```
              Question + Answer A + Answer B
                         |
         +-----------+   |   +-----------+
         | Advocates |<--+-->| Advocates |
         |  (Pro A)  |       |  (Pro B)  |
         +-----------+       +-----------+
                \               /
                 v             v
              +-------------------+
              |       Judge       |
              | Scores + Feedback |
              +-------------------+
                       |
                       v
              +-------------------+
              |    Jury Panel     |
              | (5 diverse roles) |
              +-------------------+
                       |
                       v
                Winner: A or B
```

### Two Evaluation Protocols

| | MORE (Algorithm 1) | SAMRE (Algorithm 2) |
|---|---|---|
| **Strategy** | Breadth -- k parallel advocates per answer | Depth -- iterative multi-round debate |
| **Advocates** | k=3 per side (parallel generation) | 1 per side |
| **Rounds** | Single round | Up to 5 rounds (adaptive) |
| **Stopping** | After one aggregated evaluation | Convergence detection or token budget |
| **Best for** | Fast, cost-efficient evaluation | Nuanced, complex comparisons |

### Jury Deliberation

Five persona-diverse jurors (from Appendix D.3 of the paper) independently review an anonymized debate transcript and vote. A **strict majority** (>50%) is required; ties fall back to the Judge's cumulative scores.

| Juror | Persona |
|-------|---------|
| C1 | Retired professor of ethics |
| C2 | Young environmental activist |
| C3 | Middle-aged business owner |
| C4 | Social worker in community development |
| C5 | Technology entrepreneur with AI background |

## Benchmark Results

Evaluated on [MT-Bench](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) and [AUTO-J](https://github.com/GAIR-NLP/auto-j) using `gpt-5.4-nano` as the backbone LLM.

### Agreement with Human Labels

| Dataset | Protocol | Samples | Agreement | Cohen's Kappa |
|---------|----------|---------|-----------|---------------|
| MT-Bench | MORE | 20 | **68.8%** | 0.375 |
| MT-Bench | MORE | 5 | 60.0% | 0.000 |
| MT-Bench | SAMRE | 5 | 60.0% | 0.000 |
| AUTO-J | SAMRE | 5 | **80.0%** | 0.000 |

### Cost Efficiency

| Dataset | Protocol | Samples | Mean Tokens/Eval | Total Tokens | Wall Time |
|---------|----------|---------|------------------|--------------|-----------|
| MT-Bench | MORE | 20 | ~19,051 | 381,015 | 414.9s |
| MT-Bench | MORE | 5 | ~15,911 | 79,553 | 93.5s |
| MT-Bench | SAMRE | 5 | ~15,424 | 77,120 | 123.0s |
| AUTO-J | SAMRE | 5 | ~14,648 | 73,239 | 114.6s |

> **Note:** Kappa values improve significantly with larger sample sizes. The 20-sample MT-Bench run shows kappa=0.375 (fair agreement), demonstrating that D3's adversarial debate framework produces meaningful signal even with a lightweight model.

## Project Structure

```
Project_D3/
├── main.py                    # CLI entry point (evaluate + batch subcommands)
├── config.py                  # D3Config dataclass with all framework settings
├── agents/
│   ├── advocate.py            # Advocate agent (MORE + SAMRE prompts from Appendix F)
│   ├── judge.py               # Judge agent (scoring, feedback, convergence check)
│   └── jury.py                # Jury panel (5 persona-diverse jurors, strict majority)
├── protocols/
│   ├── more.py                # MORE: Multi-Advocate One-Round Evaluation (Algorithm 1)
│   └── samre.py               # SAMRE: Single-Advocate Multi-Round Evaluation (Algorithm 2)
├── evaluation/
│   ├── benchmarks.py          # Dataset loaders (MT-Bench, AlignBench, AUTO-J)
│   ├── metrics.py             # Agreement rate, Cohen's Kappa, position bias, cost stats
│   └── runner.py              # Batch evaluation runner with optional bias measurement
├── utils/
│   ├── scoring.py             # Score gap analysis & multi-advocate aggregation
│   └── budget.py              # Token budget manager for SAMRE stopping rule
├── data/
│   ├── mt-bench/              # 3,346 pairwise samples from LMSYS
│   ├── auto-j/                # 1,391 pairwise samples from GAIR-NLP
│   └── alignbench/            # 683 single-answer samples from THUDM (Chinese)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/Project_D3.git
cd Project_D3
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-proj-your-key-here
```

### 3. Download datasets

The benchmark datasets are included under `data/`. If you need to re-download:

- **MT-Bench:** `pip install datasets` then load from `lmsys/mt_bench_human_judgments`
- **AUTO-J:** Download from [GAIR-NLP/auto-j](https://github.com/GAIR-NLP/auto-j/tree/main/data/test)
- **AlignBench:** Download from [THUDM/AlignBench](https://github.com/THUDM/AlignBench)

## Usage

### Single-pair evaluation

Compare two candidate answers to a question:

```bash
# MORE protocol (3 parallel advocates per side, single round)
python main.py evaluate \
  --protocol more \
  --question "What causes rain?" \
  --answer1 "Rain forms through the water cycle: evaporation, condensation into clouds, and precipitation when droplets grow heavy enough to fall." \
  --answer2 "Rain happens when clouds get too heavy."

# SAMRE protocol (iterative debate, up to 5 rounds)
python main.py evaluate \
  --protocol samre \
  --question "What causes rain?" \
  --answer1 "Rain forms through the water cycle..." \
  --answer2 "Rain happens when clouds get too heavy."

# Judge-only mode (no jury)
python main.py evaluate --protocol more --no-jury \
  --question "..." --answer1 "..." --answer2 "..."

# Save results to JSON
python main.py evaluate --protocol more --output-json result.json \
  --question "..." --answer1 "..." --answer2 "..."
```

### Batch benchmark evaluation

Run D3 over an entire dataset and compute aggregate metrics:

```bash
# Evaluate 20 MT-Bench samples with MORE protocol
python main.py batch --dataset mt-bench --protocol more --max-samples 20

# Evaluate AUTO-J samples with SAMRE protocol
python main.py batch --dataset auto-j --protocol samre --max-samples 10

# Measure position bias (re-runs each sample with swapped answer order)
python main.py batch --dataset mt-bench --protocol more --max-samples 10 --measure-bias

# Custom dataset path + save full report
python main.py batch --dataset mt-bench --data-path path/to/data.jsonl \
  --protocol more --output-json results/report.json
```

### Configuration flags

| Flag | Default | Description |
|------|---------|-------------|
| `--protocol` | `more` | `more` or `samre` |
| `--model` | `gpt-5.4-nano` | Backbone LLM for all agents |
| `--num-advocates` | `3` | Advocates per answer side (MORE) |
| `--max-rounds` | `5` | Maximum debate rounds (SAMRE) |
| `--token-budget` | `4096` | Token cap for budgeted stopping (SAMRE) |
| `--use-jury` / `--no-jury` | jury on | Enable/disable jury deliberation |
| `--convergence-threshold` | `0.05` | Score gap stability threshold (SAMRE) |
| `--max-samples` | all | Limit batch to first N samples |
| `--measure-bias` | off | Run position bias measurement |
| `--output-json` | none | Save results to JSON file |

## Implementation Details

### Robustness

- **Prompt injection mitigation:** All user-supplied content wrapped in `<<<...>>>` delimiters
- **Failure handling:** Advocate, aggregation, and judge failures are caught at every phase -- protocols abort or degrade gracefully instead of passing error payloads downstream
- **Parallel execution:** MORE advocates generate arguments concurrently via `ThreadPoolExecutor`
- **Per-advocate feedback:** In SAMRE, each advocate receives their own Judge feedback (not a shared copy)

