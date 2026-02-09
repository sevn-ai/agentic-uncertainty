# Agentic Uncertainty Reveals Agentic Overconfidence

<p align="center">
  <img src="paper/figures/hero_results.png" alt="Overconfidence across models and methods" width="600">
</p>

**[Link to Paper](https://arxiv.org/abs/2602.06948)**.

This repository reproduces our SWE-bench Pro experiments on:
- Pre-execution uncertainty (`exploration`)
- Post-execution uncertainty (`review`)
- Mid-execution uncertainty traces (`mid_execution`)

## Quick Start

From this directory:

```bash
cd code/agentic_uncertainty
./install.sh
```

Then set credentials:

```bash
cp .env.example .env
# edit .env and set the keys you need (see below)
```

Required in practice:
- Model API credentials (for the model you run)
- Modal credentials (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, or `modal setup`)

## Environment Variables

Common setups:

```bash
# Azure Foundry (used for many runs)
FOUNDRY_API_KEY=...
FOUNDRY_BASE_URL=...

# OpenAI-compatible models via LiteLLM
OPENAI_API_KEY=...

# Direct Anthropic
ANTHROPIC_API_KEY=...

# Gemini
GEMINI_API_KEY=...

# Modal
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
MODAL_PROFILE=...
```

## Reproduce Experiments

Use the unified runner script:

```bash
bash run_experiments.sh
```

Useful variants:

```bash
# Resume from phase 3 (uncertainty experiments onward)
bash run_experiments.sh --phase 3

# Run only one model
bash run_experiments.sh --phase 3 --model gpt
```

## Generate Tables and Figures

```bash
uv run generate-tables --results-dir results --output results/tables
uv run generate-paper-figures --cache-dir cache --ground-truth-dir data/trajectories --output-dir ../../paper/figures
```

For mid-execution analysis:

```bash
uv run python -m agentic_uncertainty.scripts.analysis.analyze_mid_execution \
  --cache-dir cache \
  --ground-truth-dir data/trajectories \
  --output-dir results/mid_execution_analysis \
  --plots --latex
```

## Notes

- Use `Modal` for trajectory generation and large experiment batches.
- Use `uv run ...` for Python commands to ensure the correct environment.
- Main artifacts live in `cache/` (per-instance outputs) and `results/` (aggregated metrics).
- This README documents the current unified interface (`run-experiment`); older config-driven interfaces are legacy.

## Citation

```bibtex
@misc{kaddour2026agenticuncertaintyrevealsagentic,
  title={Agentic Uncertainty Reveals Agentic Overconfidence},
  author={Jean Kaddour and Srijan Patel and Gb\u00e8tondji Dovonon and Leo Richter and Pasquale Minervini and Matt J. Kusner},
  year={2026},
  eprint={2602.06948},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2602.06948},
}
```
