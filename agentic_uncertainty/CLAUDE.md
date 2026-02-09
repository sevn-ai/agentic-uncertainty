# Claude Code Guidelines

## Environment Setup (CRITICAL)

### Required Environment Variables

mini-swe-agent uses python-dotenv to automatically load environment variables from:
```
~/.config/mini-swe-agent/.env
```

**One-time setup:** Copy the project's .env file to the global config location:
```bash
cp /home/jean/ac_paper/code/SWE-bench_Pro-os/mini-swe-agent/.env ~/.config/mini-swe-agent/.env
```

After this, all Python scripts will automatically have access to:
- `GEMINI_API_KEY` - For Gemini models
- `OPENAI_API_KEY` - For GPT models
- `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` - For Modal

### Common Error: "GEMINI_API_KEY not found"

This error occurs when the global config file is missing or doesn't contain the key. Fix:
```bash
cp /home/jean/ac_paper/code/SWE-bench_Pro-os/mini-swe-agent/.env ~/.config/mini-swe-agent/.env
```

### Modal Environment

- **Always use Modal** for running experiments (`--environment-class modal`). **NEVER use local Docker.**
- Modal provides faster container startup and better reliability for the SWE-bench environments.
- This applies to ALL operations: trajectory generation, evaluation, and experiments.

## Creating Instance Files for Trajectory Generation (CRITICAL)

When running trajectory generation with `--source file`, you need a JSON file with full instance objects (not just instance IDs).

### Required Instance File Format

Each instance must have these fields:
```json
{
  "instance_id": "instance_qutebrowser__qutebrowser-xxx-vyyy",
  "problem_statement": "The issue description...",
  "image_name": "jefzda/sweap-images:qutebrowser.qutebrowser-...",
  "repo_name": "app",
  "base_commit": "abc123..."
}
```

### CRITICAL: Use image_name from the SWE-bench Pro Dataset

**NEVER construct the `image_name` yourself.** The SWE-bench Pro dataset (`data/swebench_pro_dataset/test.jsonl`) already contains the correct `image_name` field for each instance. The image names are truncated to fit DockerHub's tag length limits.

### Creating Instance Files

Use the helper script to create properly formatted instance files:

```bash
cd /home/jean/ac_paper/code/agentic_uncertainty/data

# From an ID list file (creates instances_final_full.json)
python create_instances_file.py instances_final.json

# With custom output path
python create_instances_file.py instances_final.json my_instances.json

# From comma-separated IDs
python create_instances_file.py --ids "instance_NodeBB__...,instance_qutebrowser__..." output.json
```

### Available Instance Files

- `instances_final.json` / `instances_final_full.json` - 100 instances for experiments

**Always use the `_full.json` versions for trajectory generation.**

### Common Error: "RemoteError: Image build for im-xxx failed"

This error occurs when:
1. The `image_name` is incorrect (most common - you constructed it instead of using the dataset)
2. The Docker image doesn't exist on DockerHub for that instance

**To fix:** Regenerate your instances file using the script above, which uses `image_name` directly from `test.jsonl`.

## Evaluation

### Prerequisites

1. **Environment Setup:** The eval script auto-loads from `~/.config/mini-swe-agent/.env` via python-dotenv.
2. **Run from SWE-bench_Pro-os directory:** The script needs access to `dockerfiles/` and `run_scripts/`.
3. **Use the venv python:** Use the mini-swe-agent venv which has all dependencies.

### Patch File Format

The evaluation script expects a JSON list with `instance_id` and `patch` fields:

```json
[
  {"instance_id": "instance_NodeBB__NodeBB-xxx", "patch": "diff --git a/..."},
  ...
]
```

**Converting from trajectory format:** If your preds file is a dict keyed by instance_id:
```python
import json
with open('preds.json') as f:
    preds = json.load(f)
converted = [{"instance_id": k, "patch": v.get("model_patch", "")} for k, v in preds.items()]
with open('preds_converted.json', 'w') as f:
    json.dump(converted, f)
```

### Running Evaluation

NEVER use --use_local_docker - always use Modal

### Common Errors

- **"Token missing. Could not authenticate client"** - Modal tokens not found. Check `~/.config/mini-swe-agent/.env` exists and has `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
- **"No such file or directory: dockerfiles/..."** - Not running from SWE-bench_Pro-os directory.
- **All evaluations return False** - Check the evaluation output for actual errors. 0% pass rate usually means Modal auth issues.

### IMPORTANT: Use `modal` not `swerex_modal`

When running mini-swe-agent with Modal, you **MUST** use `--environment-class modal`, NOT `swerex_modal`.

**Why:** SWE-bench Pro Docker images have `/bin/bash` as their entrypoint, which conflicts with Modal's sandbox execution. The `modal` environment class calls `.entrypoint([])` to reset the entrypoint, while `swerex_modal` does not.

**Symptom of using wrong class:** `RemoteError: Image build for im-xxx failed`

```bash
# CORRECT - use modal
uv run python -m minisweagent.run.extra.run_batch --environment-class modal ...

# WRONG - do not use swerex_modal for SWE-bench Pro
uv run python -m minisweagent.run.extra.run_batch --environment-class swerex_modal ...
```

## Model

- **Always use `gpt-5.2-codex`** as the model. Never use Claude models here.

## Running Tests

### Unified CLI (Preferred)

Single-instance exploration test:
```bash
uv run run-experiment --agents exploration --instance-ids data/instances_1.json --ground-truth data/trajectories/gpt-5.2-codex/evaluation/eval_results.json --num-samples 1 --output-dir results/test --parallel 1 --environment-class modal --model gpt-5.2-codex --exploration-methods exploration_direct
```

Single-instance review test:
```bash
uv run run-experiment --agents review --instance-ids data/instances_1.json --traj-dir data/trajectories/gpt-5.2-codex --ground-truth data/trajectories/gpt-5.2-codex/evaluation/eval_results.json --num-samples 1 --output-dir results/review_test --parallel 1 --environment-class modal --model gpt-5.2-codex --review-methods direct
```

Run both exploration and review on the same instances:
```bash
uv run run-experiment --agents exploration review --instance-ids data/instances_1.json --traj-dir data/trajectories/gpt-5.2-codex --ground-truth data/trajectories/gpt-5.2-codex/evaluation/eval_results.json --num-samples 1 --output-dir results/test --parallel 1 --environment-class modal --model gpt-5.2-codex --exploration-methods exploration_direct --review-methods direct
```
