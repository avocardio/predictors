# Predictors Framework

An automated framework for discovering and training novel prediction tasks using reasoning models and cloud compute.

## Overview

This framework automatically:
1. Uses GPT-4o/o3 to discover creative cross-domain prediction tasks
2. Generates experiment code and dataloaders
3. Downloads data from API-friendly sources (Kaggle, HuggingFace, etc.)
4. Trains transformer models on VAST AI GPUs
5. Tracks all experiments to avoid duplicates

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys in `config.yaml`:
```yaml
openai_api_key: "your-key"
vast_api_key: "your-key"
```

Or set environment variables:
```bash
export OPENAI_API_KEY="your-key"
export VAST_API_KEY="your-key"
```

3. (Optional) Configure Kaggle API:
```bash
# Place kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### Run a new experiment
```bash
python orchestrator.py --run
```

### Run locally (no cloud)
```bash
python orchestrator.py --run --local
```

### List all experiments
```bash
python orchestrator.py --list
```

### Weekly automation (crontab)
```bash
0 0 * * 0 cd /path/to/predictors && python orchestrator.py --run
```

## Architecture

- `task_discovery.py` - GPT-4o reasoning for creative task generation
- `data_acquisition.py` - API clients for data sources
- `base_model.py` - PyTorch Lightning transformer
- `experiment_generator.py` - Creates experiment folders
- `vast_runner.py` - VAST AI GPU deployment
- `orchestrator.py` - Main pipeline controller
- `history.json` - Tracks all attempted tasks

## Example Tasks

The system discovers tasks like:
- Predicting time of day from ambient sounds
- Inferring artist mood from painting colors
- Correlating weather patterns with social sentiment
- Predicting building age from architectural features

Each experiment is self-contained in `experiments/[timestamp]_[task_name]/`