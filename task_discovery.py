import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List
try:
    from openai import OpenAI
    USE_OPENAI = True
except ImportError:
    from anthropic import Anthropic
    USE_OPENAI = False


class TaskDiscovery:
    def __init__(self, api_key: str, use_openai: bool = False):
        if use_openai or USE_OPENAI:
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4-turbo-preview"  # or "o1-preview" for reasoning
            self.use_openai = True
        else:
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-opus-4-1-20250805"  # Latest with extended thinking + web search
            self.use_openai = False
        self.history_file = Path("history.json")
    
    def load_history(self) -> Dict:
        """Load previous tasks from history"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {"tasks": [], "metadata": {"created_at": datetime.now().isoformat(), "total_experiments": 0}}
    
    def discover_novel_task(self) -> Dict:
        """Use Claude API to discover a novel prediction task"""
        history = self.load_history()
        previous_tasks = [task.get('name', '') for task in history.get('tasks', [])]
        
        prompt = f"""You are an automated ML researcher. Your job is to discover a completely novel prediction task that combines unexpected data sources.

PREVIOUS TASKS (avoid these):
{chr(10).join(f"- {task}" for task in previous_tasks[-10:])}

DISCOVER A NEW TASK:
1. Think of a creative cross-domain prediction:
   - Weather patterns → social media sentiment
   - Music features → stock market movements  
   - Art styles → historical economic indicators
   - Writing patterns → personality traits
   - Image colors → cultural characteristics

2. Find datasets available via APIs (Kaggle, HuggingFace, OpenML, direct URLs)
3. Ensure it's transformer-solvable and truly novel

Return ONLY valid JSON with this structure:
{{
  "name": "short_descriptive_name",
  "description": "detailed explanation of what and why",
  "input_description": "what the model takes as input",
  "output_description": "what the model predicts",
  "data_sources": [
    {{"type": "kaggle", "identifier": "username/dataset-name", "description": "what this provides"}},
    {{"type": "huggingface", "identifier": "dataset-name", "description": "what this provides"}}
  ],
  "task_type": "regression",
  "evaluation_metric": "rmse",
  "expected_baseline": "random baseline performance estimate"
}}

Focus on genuinely interesting correlations that haven't been explored before!"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            
            # Find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                task_json = json.loads(json_match.group())
                task_json["created_at"] = datetime.now().isoformat()
                task_json["id"] = f"task_{len(previous_tasks):03d}"
                return task_json
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            print(f"Error in task discovery: {e}")
            # Fallback to a simple task
            return self._fallback_task(len(previous_tasks))
    
    def _fallback_task(self, task_num: int) -> Dict:
        """Fallback task if API fails"""
        fallback_tasks = [
            {
                "name": "weather_sentiment_prediction",
                "description": "Predict social media sentiment from weather patterns",
                "input_description": "Weather data (temperature, humidity, pressure)",
                "output_description": "Social media sentiment score",
                "data_sources": [
                    {"type": "kaggle", "identifier": "berkeleyearth/climate-change-earth-surface-temperature-data", "description": "weather data"},
                    {"type": "huggingface", "identifier": "tweet_eval", "description": "sentiment data"}
                ],
                "task_type": "regression",
                "evaluation_metric": "rmse",
                "expected_baseline": "0.5"
            }
        ]
        
        task = fallback_tasks[task_num % len(fallback_tasks)]
        task["created_at"] = datetime.now().isoformat()
        task["id"] = f"fallback_{task_num:03d}"
        return task
    
    def save_task(self, task: Dict):
        """Save completed task to history"""
        history = self.load_history()
        
        # Ensure metadata exists
        if "metadata" not in history:
            history["metadata"] = {
                "created_at": datetime.now().isoformat(),
                "total_experiments": 0,
                "last_updated": datetime.now().isoformat()
            }
        
        history["tasks"].append(task)
        history["metadata"]["total_experiments"] = len(history["tasks"])
        history["metadata"]["last_updated"] = datetime.now().isoformat()
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Saved task '{task['name']}' to history")