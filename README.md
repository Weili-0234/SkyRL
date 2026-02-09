# SkyRL RL Training Reproduction: DP=4 Default vs TR(atw=0.1)

## Overview
This directory provides a reproducible workflow for running SkyRL RL training with ThunderAgent on a SLURM cluster, and comparing rollout throughput in **tokens/sec** for:

- `dp4_default`: ThunderAgent router `default`
- `dp4_tr_atw01`: ThunderAgent router `tr` with `acting_token_weight=0.1`

The workflow is designed for the deployment style where this SkyRL repo is used as an example under:

- `ThunderAgent-wl/examples/rl_training/`

## Reproduction Target
The goal is to reproduce the **per-step rollout throughput (tokens/sec)** comparison under:

- Model: `Qwen/Qwen3-14B`
- Inference layout: DP=4, TP=1 (4 vLLM engines)
- Training layout: FSDP2, SP=4, non-colocated (4 training GPUs)
- Data: `SumanthRH/SWE-Gym-Subset`
- Batch config: `train_batch_size=99`, `n_samples_per_prompt=4` (396 trajectories/step)

Throughput definition is exactly aligned with `analyze_all.py`:

- `tokens/sec = (396 * avg_response_length) / generate_duration_seconds`

## Expected Directory Layout
This README assumes a layout like:

```text
<THUNDERAGENT_ROOT>/
├── examples/
│   └── rl_training/
│       └── SkyRL/                      # this repo
└── ThunderAgent/                       # python package source
```

In this layout:

- `SKYRL_HOST_DIR=<THUNDERAGENT_ROOT>/examples/rl_training/SkyRL`
- `THUNDERAGENT_HOST_DIR=<THUNDERAGENT_ROOT>`

If your layout differs, set these paths explicitly (see "Required Customization").

## Prerequisites
- SLURM cluster access (1 node, 8 GPUs, tested with H100 80GB)
- Docker available on compute nodes
- Permission to run `sudo` for Docker daemon configuration (`/etc/docker/daemon.json`)
- Python 3.10+ on the submission host
- WANDB API key (required)
- HuggingFace token (recommended for model/dataset/image pulls)

## Required Customization
Before running, customize the following.

1. `api-keys.sh` in SkyRL root:
```bash
export WANDB_API_KEY="<your_wandb_key>"
# optional but recommended
export HF_TOKEN="<your_hf_token>"
```

2. SBATCH header fields in:
- `sbatch_repro_dp4_default.sh`
- `sbatch_repro_dp4_tr_atw01.sh`

Typical fields to edit for your cluster:
- `#SBATCH --partition=...`
- `#SBATCH --time=...`
- `#SBATCH --cpus-per-task=...`
- `#SBATCH --gpus=...`
- optional: `#SBATCH --account=...`, `#SBATCH --qos=...`, `#SBATCH --exclude=...`

3. Runtime path/image variables (export before submission):
```bash
export SKYRL_HOST_DIR="<path-to-SkyRL>"
export THUNDERAGENT_HOST_DIR="<path-to-ThunderAgent-wl-root>"
export TRAINING_DATA_SYNC_DEST="<where-to-sync-training-artifacts>"
export DOCKER_IMAGE="novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8"
```

If not set, defaults are inferred from the expected layout above.

## Setup
From your SkyRL directory:

```bash
cd <THUNDERAGENT_ROOT>/examples/rl_training/SkyRL

# Optional: verify key files
ls launch_profiled_training_14b_dp4.sh
ls skyrl-train/examples/mini_swe_agent/run_mini_swe_14B_dp4.sh
```

Prepare credentials:

```bash
cat > api-keys.sh <<'SH'
export WANDB_API_KEY="<your_wandb_key>"
export HF_TOKEN="<your_hf_token>"
SH
chmod 600 api-keys.sh
```

## Reproduction
### Option A: Submit both jobs with one command
```bash
cd <THUNDERAGENT_ROOT>/examples/rl_training/SkyRL

export SKYRL_HOST_DIR="$(pwd)"
export THUNDERAGENT_HOST_DIR="$(cd ../../.. && pwd)"
export TRAINING_DATA_SYNC_DEST="$(pwd)/training_data"

bash submit_repro_dp4_default_vs_tr_atw01.sh
```

### Option B: Submit jobs separately
```bash
cd <THUNDERAGENT_ROOT>/examples/rl_training/SkyRL

sbatch --export=ALL sbatch_repro_dp4_default.sh
sbatch --export=ALL sbatch_repro_dp4_tr_atw01.sh
```

## Monitoring
Check job states:

```bash
squeue -u "$USER" --format="%.10i %.20j %.8T %.10M %.20N"
```

Check logs:

```bash
tail -n 40 slurm-<JOBID>.out
tail -n 60 slurm-<JOBID>.err
```

Useful rollout markers:

```bash
grep "Started: 'step'" slurm-<JOBID>.err
grep "Finished: 'generate'" slurm-<JOBID>.err
grep "avg_response_length" slurm-<JOBID>.err
```

## Throughput Analysis (tokens/sec only)
After both jobs complete at least one rollout step:

```bash
python3 analyze_dp4_tokens_compare.py \
  --default-job <DEFAULT_JOB_ID> \
  --tr-job <TR_JOB_ID>
```

The script prints a focused markdown table:

- one row per rollout step
- `dp4_default tokens/sec`
- `dp4_tr_atw01 tokens/sec`
- delta and delta percentage
- average over paired completed steps

This script intentionally reports only throughput (tokens/sec), not the full metric set from `analyze_all.py`.

## New Repro Scripts in This Directory
- `sbatch_repro_dp4_default.sh`: explicit DP=4 default-router job
- `sbatch_repro_dp4_tr_atw01.sh`: explicit DP=4 TR job with `acting_token_weight=0.1`
- `scripts/repro/run_dp4_repro_job.sh`: shared SLURM job body used by both scripts
- `submit_repro_dp4_default_vs_tr_atw01.sh`: submit both jobs and print job IDs
- `analyze_dp4_tokens_compare.py`: focused per-step tokens/sec comparison table

## Notes
- Existing legacy launch/analysis scripts are left unchanged.
- `sbatch_repro_dp4_tr_atw01.sh` is explicit by name so the TR setting is unambiguous.
- Training artifacts are synced to `TRAINING_DATA_SYNC_DEST/job_<JOBID>_dp4_<tag>/`.
