import openai
import json
import random
from typing import Dict, List, Optional
from datetime import datetime
import os


class TaskDiscovery:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.history_file = "history.json"
        self.load_history()
    
    def load_history(self):
        try:
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.history = {"tasks": [], "metadata": {}}
    
    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_creative_task(self) -> Dict:
        tried_tasks = "\n".join([f"- {task['name']}: {task['description']}" 
                                for task in self.history.get('tasks', [])])
        
        prompt = f"""You are a creative AI researcher discovering novel prediction tasks.
Your goal is to find interesting, non-obvious prediction tasks that combine data from different sources.

Previous tasks (DO NOT repeat these):
{tried_tasks if tried_tasks else "None yet"}

Requirements:
1. The task must use API-accessible data (Kaggle, HuggingFace, OpenML, or direct URLs)
2. Think about cross-domain correlations (e.g., weather patterns predicting social media sentiment)
3. Consider temporal, spatial, or abstract patterns
4. The data should be manageable (< 1GB total)
5. The task should be solvable with a transformer model

Generate a unique prediction task with the following structure:
{{
    "name": "short_descriptive_name",
    "description": "What are we predicting and why it's interesting",
    "input_description": "What the model takes as input",
    "output_description": "What the model predicts",
    "data_sources": [
        {{
            "type": "kaggle|huggingface|openml|url",
            "identifier": "dataset_id or url",
            "description": "what data this provides"
        }}
    ],
    "task_type": "regression|classification|sequence",
    "preprocessing_hints": "How to combine/process the data",
    "evaluation_metric": "What metric to use",
    "expected_baseline": "What performance a simple model might achieve"
}}

Be creative! Think about:
- Predicting art style from weather data when painted
- Inferring time of day from bird sounds
- Predicting restaurant success from street view images
- Correlating music tempo with stock market volatility
- Predicting author age from writing style
- Inferring building age from architectural features

Return ONLY valid JSON."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a creative ML researcher. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=1000
        )
        
        task_json = response.choices[0].message.content
        task = json.loads(task_json)
        
        # Add metadata
        task["id"] = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task["created_at"] = datetime.now().isoformat()
        task["status"] = "discovered"
        
        # Check for similarity with existing tasks
        if self._is_too_similar(task):
            print(f"Task too similar to existing one, retrying...")
            return self.get_creative_task()
        
        return task
    
    def _is_too_similar(self, new_task: Dict) -> bool:
        for existing_task in self.history.get('tasks', []):
            if self._calculate_similarity(new_task['name'], existing_task['name']) > 0.7:
                return True
            if self._calculate_similarity(new_task['description'], existing_task['description']) > 0.7:
                return True
        return False
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)
    
    def record_task(self, task: Dict):
        self.history['tasks'].append(task)
        self.save_history()
    
    def generate_dataloader_code(self, task: Dict) -> str:
        prompt = f"""Generate a PyTorch dataloader for this task:
Task: {task['name']}
Description: {task['description']}
Input: {task['input_description']}
Output: {task['output_description']}
Data sources: {json.dumps(task['data_sources'], indent=2)}
Task type: {task['task_type']}
Preprocessing: {task['preprocessing_hints']}

Generate a complete dataloader.py file that:
1. Downloads/loads data from the specified sources
2. Preprocesses according to the hints
3. Returns dictionaries with 'input', 'target', and optionally 'mask' keys
4. Includes train/val/test splits
5. Is compatible with the base transformer model

Return ONLY the Python code, no explanations."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert ML engineer. Return only Python code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def analyze_task_dimensions(self, task: Dict) -> Dict:
        """Analyze task to determine model dimensions"""
        prompt = f"""Analyze this task and determine transformer model dimensions:
Task: {json.dumps(task, indent=2)}

Return JSON with:
{{
    "input_dim": <integer>,
    "output_dim": <integer>,
    "suggested_d_model": <256|512|768>,
    "suggested_layers": <4|6|8>,
    "estimated_training_steps": <integer>
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        return json.loads(response.choices[0].message.content)