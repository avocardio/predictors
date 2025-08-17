# Run Complete Pipeline

Execute the full predictors pipeline: discover task, generate experiment, and run training.

## Instructions

You are running the complete Predictors pipeline. Execute these steps in order:

### Step 1: Discover Task
1. Check `history.json` for previous tasks
2. Think creatively about novel cross-domain prediction tasks
3. Create `experiments/next_task.json` with the task specification
4. Ensure the task is genuinely novel and feasible

### Step 2: Generate Experiment
1. Read the task from `experiments/next_task.json`
2. Create experiment folder: `experiments/YYYYMMDD_HHMMSS_TASKNAME/`
3. Copy `base_model.py` to experiment folder as `model.py`
4. Generate `dataloader.py` with proper data handling
5. Generate `train.py` with appropriate configuration
6. Copy task specification to experiment folder

### Step 3: Run Training
1. Change to the experiment directory
2. Install any missing requirements: `pip install -r ../requirements.txt`
3. Execute training: `python train.py`
4. Monitor progress and handle any errors
5. Ensure `results.json` is created with final metrics

### Step 4: Update History
1. Read current `history.json`
2. Add the completed task with results
3. Include timestamp and experiment path
4. Save updated history

### Step 5: Cleanup
1. Run cleanup on the experiment folder
2. Remove large unnecessary files
3. Keep only essential results and code

### Error Handling
- If data download fails, try alternative sources
- If training crashes, adjust model size or learning rate
- Log all errors and continue with partial results if possible

### Success Criteria
- Task is novel and interesting
- Data downloads successfully
- Training completes without major errors
- Results are saved properly
- History is updated

Execute each step carefully and report progress along the way.