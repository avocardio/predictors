# Generate Experiment

Generate a complete experiment folder for a discovered task.

## Arguments
- `$TASK_NAME`: Name of the task to generate experiment for

## Instructions

You are generating a complete experiment for the specified task. Read the task specification from `experiments/next_task.json`.

### Experiment Structure
Create a new experiment folder: `experiments/YYYYMMDD_HHMMSS_$TASK_NAME/`

### Required Files

#### 1. Copy Base Model
Copy `base_model.py` to the experiment folder as `model.py`

#### 2. Generate dataloader.py
Create a complete PyTorch dataloader that:
- Downloads data from the specified API sources
- Handles data preprocessing and cleaning
- Implements proper train/val/test splits (70/15/15)
- Returns a `get_dataloaders(batch_size=32, num_workers=4)` function
- Each batch should be a dict with keys: `{'input': tensor, 'target': tensor, 'mask': optional_tensor}`
- Use appropriate libraries: `datasets` for HuggingFace, `kaggle` API, `requests` for URLs

#### 3. Generate train.py  
Create a complete training script that:
- Imports `BaseTransformer` from `model.py`
- Imports `get_dataloaders` from `dataloader.py`
- Configures model dimensions based on data
- Uses PyTorch Lightning Trainer with GPU support
- Includes callbacks: ModelCheckpoint, EarlyStopping
- Saves final results to `results.json`
- Handles errors gracefully

#### 4. Copy task.json
Copy the task specification to the experiment folder

### Model Configuration
Analyze the task to determine:
- `input_dim`: Based on processed input features
- `output_dim`: Based on prediction target
- `d_model`: 256 for simple tasks, 512 for complex
- `num_layers`: 4-8 depending on task complexity
- `task_type`: 'regression', 'classification', or 'sequence'

### Data Handling
- Cache downloaded data in `data/` subfolder
- Implement proper error handling for data download failures
- Add data validation and size checks
- Use appropriate preprocessing for the task type

Only work within the experiment folder. Do not modify files outside of it.