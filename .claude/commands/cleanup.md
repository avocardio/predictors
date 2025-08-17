# Cleanup Experiment

Clean up an experiment folder to save space and keep only essential files.

## Arguments
- `$EXPERIMENT_PATH`: Path to the experiment folder to clean up

## Instructions

You are cleaning up a completed experiment to save disk space while preserving essential results.

### Files to Keep
- `task.json` - Task specification
- `results.json` - Final training results
- `model.py` - Model architecture
- `train.py` - Training script
- `dataloader.py` - Data loading code
- Best model checkpoint (usually `checkpoints/best-*.ckpt`)

### Files/Folders to Remove or Compress
1. **Large Data Files**: Remove raw data files > 100MB from `data/` folder
2. **Extra Checkpoints**: Keep only the best checkpoint, remove others
3. **Cache Folders**: Remove `__pycache__/`, `.pytest_cache/`, etc.
4. **Log Files**: Compress large log files if > 10MB
5. **Temporary Files**: Remove any `.tmp`, `.temp`, or similar files

### Actions to Take
```bash
# Navigate to experiment folder
cd $EXPERIMENT_PATH

# Remove large data files but keep metadata
find data/ -size +100M -type f -delete 2>/dev/null || true

# Keep only best checkpoint
if [ -d "checkpoints" ]; then
    # Find the best checkpoint (usually has 'best' in name)
    best_ckpt=$(ls checkpoints/ | grep -i best | head -1)
    if [ -n "$best_ckpt" ]; then
        mkdir -p temp_checkpoints
        cp "checkpoints/$best_ckpt" temp_checkpoints/
        rm -rf checkpoints
        mv temp_checkpoints checkpoints
    fi
fi

# Remove cache directories
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true

# Compress large log files
find . -name "*.log" -size +10M -exec gzip {} \; 2>/dev/null || true

# Remove temporary files
find . -name "*.tmp" -o -name "*.temp" -delete 2>/dev/null || true
```

### Summary Report
After cleanup, provide a summary:
- Files removed and space saved
- Files preserved
- Any issues encountered

Do not remove the core experiment files needed to understand and potentially reproduce the results.