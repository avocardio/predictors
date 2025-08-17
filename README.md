# Predictors Framework

An automated framework that discovers and trains novel prediction tasks using Claude Code and cloud GPUs.

## Overview

This framework runs entirely on cloud GPU servers and uses Claude Code instances to:
1. **Discover** creative cross-domain prediction tasks
2. **Generate** complete experiment code and dataloaders
3. **Train** transformer models on GPU hardware
4. **Track** all experiments to avoid duplicates

The key innovation is using Claude Code as the reasoning engine instead of simple API calls, allowing for complex file operations and creative problem-solving.

## Architecture

```
Framework runs on RunPod GPU server:
┌─────────────────────────────────────┐
│ GPU Server (/workspace/predictors)  │
├─────────────────────────────────────┤
│ startup_script.py                   │ ← Installs Claude Code + deps
│ CLAUDE.md                          │ ← Instructions for Claude
│ .claude/commands/                   │ ← Slash commands
│   ├── discover.md                  │ ← /discover task
│   ├── generate.md                  │ ← /generate experiment  
│   ├── run.md                       │ ← /run full pipeline
│   └── cleanup.md                   │ ← /cleanup experiment
│ experiments/                       │ ← Generated experiments
│ history.json                       │ ← Task tracking
└─────────────────────────────────────┘
```

## Setup

### 1. Configure APIs
Set environment variables:
```bash
export RUNPOD_API_KEY="your-runpod-key"
export ANTHROPIC_API_KEY="your-claude-key"
export KAGGLE_KEY="your-kaggle-key"  # optional
```

### 2. Launch GPU Server
```python
from runpod_runner import RunPodRunner

runner = RunPodRunner()
pod_id = runner.create_pod(gpu_type="RTX 4090")
print(f"GPU server started: {pod_id}")
```

The startup script will:
- Install Claude Code and dependencies
- Clone this repository
- Launch Claude Code with `--dangerously-skip-permissions`

### 3. Use Claude Code Commands

Once Claude Code is running on the GPU server, use these slash commands:

#### Discover a new task:
```
/discover
```

#### Generate and run complete pipeline:
```
/run
```

#### Generate experiment for a specific task:
```
/generate task_name
```

#### Clean up an experiment:
```
/cleanup experiments/20250817_143022_music_mood_prediction
```

## How It Works

### Task Discovery
Claude Code reads `history.json` and thinks creatively about novel prediction tasks like:
- Predicting weather from social media image colors
- Inferring building age from architectural features  
- Correlating music tempo with stock market patterns
- Predicting artist mood from painting brush strokes

### Experiment Generation
For each task, Claude Code:
1. Downloads data from APIs (HuggingFace, Kaggle, etc.)
2. Generates a custom PyTorch dataloader
3. Configures the transformer model appropriately
4. Creates a complete training script

### Training & Results
- Runs entirely on GPU using PyTorch Lightning
- Saves results to `results.json`
- Updates `history.json` to avoid duplicates
- Cleans up large files automatically

## Example Workflow

1. **Start GPU server**: `python -c "from runpod_runner import RunPodRunner; RunPodRunner().create_pod()"`

2. **SSH to server** (or use RunPod web terminal)

3. **In Claude Code**: `/run` 

4. **Claude discovers**: "Predicting music decade from album cover colors"

5. **Claude generates**: Complete experiment with Spotify API + image processing

6. **Claude trains**: Transformer model on GPU for 2 hours

7. **Results saved**: Accuracy metrics and model checkpoints

8. **History updated**: Task marked complete to avoid future duplicates

## Costs

Approximate costs using RunPod:
- RTX 4090: $0.44/hour
- A100 40GB: $1.09/hour  
- H100 80GB: $2.99/hour

Most experiments complete in 1-3 hours = $0.44-$9 per novel prediction task.

## Weekly Automation

For continuous discovery, set up a weekly trigger:
```python
# Weekly cron job or GitHub Actions
runner = RunPodRunner()
pod_id = runner.create_pod()
# Claude Code will auto-run /run command
runner.wait_for_completion(pod_id)
runner.terminate_pod(pod_id)
```

## Files

- `runpod_runner.py` - GPU server management
- `startup_script.py` - Server initialization  
- `CLAUDE.md` - Instructions for Claude Code
- `.claude/commands/` - Slash commands
- `base_model.py` - PyTorch Lightning transformer
- `data_acquisition.py` - Data download utilities
- `experiment_generator.py` - Experiment scaffolding

The framework is designed to be fully autonomous - just launch a GPU server and let Claude Code discover and train novel prediction tasks!