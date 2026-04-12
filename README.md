# D3: Debate, Deliberate, Decide

A Cost-Aware Adversarial Framework for Reliable and Interpretable LLM Evaluation.

**Paper:** [arXiv:2410.04663](https://arxiv.org/abs/2410.04663)

## Overview

D3 orchestrates role-specialized agents (advocates, a judge, and an optional jury) to evaluate LLM outputs through structured debate. It implements two protocols:

- **MORE** (Multi-Advocate One-Round Evaluation) — k parallel advocates amplify signal via diverse advocacy
- **SAMRE** (Single-Advocate Multi-Round Evaluation) — iterative argument refinement under an explicit token budget with convergence checks

## Project Structure

```
Project_D3/
├── main.py                  # Entry point
├── config.py                # Framework configuration
├── agents/
│   ├── advocate.py          # Advocate agents (pro/con)
│   ├── judge.py             # Judge agent
│   └── jury.py              # Optional jury agent
├── protocols/
│   ├── more.py              # MORE protocol
│   └── samre.py             # SAMRE protocol
├── utils/
│   ├── scoring.py           # Score aggregation & gap analysis
│   └── budget.py            # Token budget manager
├── evaluation/
│   └── benchmarks.py        # Dataset loaders (MT-Bench, AlignBench, AUTO-J)
├── data/                    # Dataset files
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

## Usage

```bash
# Run with MORE protocol (default)
python main.py --protocol more --dataset mt-bench --num-advocates 3

# Run with SAMRE protocol
python main.py --protocol samre --dataset mt-bench --token-budget 4096

# Enable jury
python main.py --protocol more --use-jury
```
