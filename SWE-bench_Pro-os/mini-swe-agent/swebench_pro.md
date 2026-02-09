# Running mini-extra run-batch

This guide shows you how to run batch evaluations using `mini-extra run-batch`.

## Quick Start

Minimal command to run batch evaluation:

```bash
mini-extra run-batch \
    --instances-path instances.json \
    --output-dir results \
    --model gpt-4
```

## Complete Example

Full-featured command with all common options:

```bash
mini-extra run-batch \
    --config config/default.yaml \
    --output-dir sweagent_results/test/claude-sonnet-4-5 \
    --num-workers 50 \
    --random-delay-multiplier 1 \
    --source file \
    --instances-path sweagent_wrapper_configs/instances_test_file.yaml \
    --no-shuffle \
    --deployment-type modal \
    --deployment-install-pipx \
    --deployment-startup-timeout 900 \
    --per-instance-call-limit 250 \
    --per-instance-cost-limit 0 \
    --total-cost-limit 0 \
    --model anthropic/claude-haiku-4-5 \
    --model-api-base https://litellm.ml-serving-internal.scale.com/v1 \
    --model-api-key $OPENAI_API_KEY \
    --model-temperature 0.0
```

## All Available CLI Options

### Instance Loading
- `--instances-path PATH` - Path to instances file (JSON/JSONL/YAML)
- `--source {file,swebench,huggingface}` - Instance source type
- `--subset {lite,verified,full,multimodal,multilingual}` - SWE-bench subset
- `--split {dev,test}` - Dataset split
- `--dataset-name NAME` - HuggingFace dataset name

### Instance Filtering
- `--filter REGEX` - Filter instance IDs by regex (default: ".*")
- `--slice SLICE` - Slice specification (e.g., "0:10" or "::2")
- `--shuffle` / `--no-shuffle` - Enable/disable shuffling (default: no shuffle)

### Basic Options
- `-o, --output, --output-dir DIR` - Output directory
- `-w, --workers, --num-workers N` - Number of parallel workers
- `--config PATH` - Agent configuration file

### Model Options
- `-m, --model NAME` - Model name
- `--model-class CLASS` - Model class
- `--model-api-base URL` - Model API base URL
- `--model-api-key KEY` - Model API key
- `--model-temperature FLOAT` - Sampling temperature
- `--model-top-p FLOAT` - Top-p sampling

### Model Limits
- `--per-instance-call-limit N` - Max API calls per instance (0=unlimited)
- `--per-instance-cost-limit FLOAT` - Max cost per instance (0=unlimited)
- `--total-cost-limit FLOAT` - Max total cost (0=unlimited)

### Environment Options
- `--environment-class {docker,singularity,local,modal}` - Environment type
- `--deployment-type {modal}` - Deployment type (sets environment to modal)
- `--deployment-install-pipx` / `--no-deployment-install-pipx` - Install pipx in Modal deployment
- `--deployment-startup-timeout SECONDS` - Modal deployment startup timeout (default: 600s)

### Advanced Options
- `--redo-existing` / `--no-redo-existing` - Re-run existing trajectories
- `--raise-exceptions` / `--no-raise-exceptions` - Stop on first error
- `--random-delay-multiplier FLOAT` - Startup delay multiplier

## Common Use Cases

### 1. Run from Custom Instances File
```bash
mini-extra run-batch \
    --instances-path instances.json \
    --output-dir results \
    --model gpt-4
```

### 2. Run SWE-bench Subset
```bash
mini-extra run-batch \
    --source swebench \
    --subset lite \
    --split dev \
    --slice 0:10 \
    --output-dir results \
    --workers 4 \
    --model anthropic/claude-3-5-sonnet-20241022
```

### 3. Run with Modal Deployment
```bash
mini-extra run-batch \
    --config config/default.yaml \
    --instances-path my_instances.yaml \
    --output-dir results/experiment \
    --workers 10 \
    --model anthropic/claude-haiku-4-5 \
    --model-api-base https://api.example.com/v1 \
    --model-api-key $API_KEY \
    --deployment-type modal \
    --deployment-startup-timeout 900
```

### 4. Run with Cost and Call Limits
```bash
mini-extra run-batch \
    --instances-path instances.json \
    --output-dir results \
    --model anthropic/claude-haiku-4-5 \
    --model-temperature 0.5 \
    --per-instance-call-limit 100 \
    --per-instance-cost-limit 1.0 \
    --total-cost-limit 50.0
```

## Testing Your Command

Before running a full batch, test with a small slice:

```bash
# Test with just the first instance
mini-extra run-batch \
    --config config/default.yaml \
    --output-dir test_results \
    --source file \
    --instances-path sweagent_wrapper_configs/instances_test_file.yaml \
    --slice 0:1 \
    --model anthropic/claude-haiku-4-5 \
    --model-api-base https://litellm.ml-serving-internal.scale.com/v1 \
    --model-api-key $OPENAI_API_KEY
```

## Important Conventions

### Boolean Flags
Use `--flag` to enable, `--no-flag` to disable (NOT `--flag False`):

```bash
# Enable shuffle
--shuffle

# Disable shuffle (default)
--no-shuffle

# Enable redo existing
--redo-existing

# Disable redo existing (default)
--no-redo-existing
```

### Naming Conventions
- Use **dashes**, not underscores: `--output-dir` not `--output_dir`
- All options are **flat** (no dots): `--model` not `--agent.model.name`
- Many options have **short aliases**: `-o`, `-w`, `-m`

### Environment Variables

Use environment variables for sensitive data like API keys:

```bash
--model-api-key $OPENAI_API_KEY
--model-api-key $ANTHROPIC_API_KEY
```

## Getting Help

View all available options:

```bash
mini-extra run-batch --help
```

The help is organized into these sections:
- **Instance Loading** - Load instances from files or datasets
- **Instance Filtering** - Filter, slice, and shuffle instances
- **Basic Options** - Core settings (output dir, workers, config)
- **Model Options** - Model selection and API configuration
- **Model Limits** - Call and cost limits
- **Environment Options** - Docker, Singularity, Local, or Modal
- **Advanced Options** - Retry behavior and timing

## Option Reference Summary

| Category | Common Options |
|----------|---------------|
| **Required** | `--instances-path` or `--source`, `--output-dir`, `--model` |
| **Instance Sources** | `--source {file,swebench,huggingface}` |
| **Filtering** | `--filter`, `--slice`, `--shuffle` |
| **Parallelization** | `--num-workers` (or `-w`) |
| **Model Config** | `--model-api-base`, `--model-api-key`, `--model-temperature` |
| **Limits** | `--per-instance-call-limit`, `--total-cost-limit` |
| **Deployment** | `--deployment-type modal`, `--deployment-startup-timeout` |

