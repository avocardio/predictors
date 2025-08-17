# Discover New Prediction Task

Discover a novel, creative prediction task that combines data from different sources.

## Instructions

You are discovering a new prediction task for the Predictors framework. Follow these guidelines:

### Check History First
- Read `history.json` to see what tasks have been tried before
- Avoid duplicating existing tasks or similar concepts

### Creative Task Requirements
Think about cross-domain correlations that are non-obvious but potentially meaningful:
- Predicting weather patterns from social media image color palettes
- Inferring music era from album cover design elements  
- Predicting restaurant success from street view architectural features
- Correlating news sentiment with cryptocurrency volatility patterns
- Predicting artist mood from painting brush stroke patterns
- Inferring time of day from ambient sound recordings

### Technical Constraints
The task must:
1. Use API-accessible data sources (HuggingFace, Kaggle, OpenML, direct URLs)
2. Be solvable with transformer architecture
3. Have total data size < 1GB 
4. Provide clear input/output specifications
5. Include evaluation metrics

### Output Format
Create a JSON file at `experiments/next_task.json` with this structure:
```json
{
  "name": "short_descriptive_name",
  "description": "What we're predicting and why it's interesting",
  "input_description": "What the model takes as input",
  "output_description": "What the model predicts",
  "data_sources": [
    {
      "type": "kaggle|huggingface|openml|url", 
      "identifier": "dataset_id or URL",
      "description": "what this data provides"
    }
  ],
  "task_type": "regression|classification|sequence",
  "evaluation_metric": "accuracy|mse|f1|etc",
  "expected_baseline": "rough performance estimate",
  "preprocessing_hints": "how to combine/process the data"
}
```

Be genuinely creative and think outside the box!