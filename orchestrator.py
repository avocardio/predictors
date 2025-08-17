#!/usr/bin/env python3
import os
import sys
import json
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from task_discovery import TaskDiscovery
from experiment_generator import ExperimentGenerator
from vast_runner import VastRunner


class Orchestrator:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.history_file = "history.json"
        
    def _load_config(self, config_path: str) -> dict:
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Create default config
            default_config = {
                "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                "vast_api_key": os.getenv("VAST_API_KEY", ""),
                "kaggle_configured": Path("~/.kaggle/kaggle.json").expanduser().exists(),
                "model": "gpt-4o",
                "min_gpu_ram": 16,
                "max_price_per_hour": 1.0,
                "max_training_hours": 3,
                "run_on_vast": True,
                "local_only": False
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            print(f"Created default config at {config_path}")
            print("Please update with your API keys")
            return default_config
    
    def run_pipeline(self, force_local: bool = False):
        """Main pipeline execution"""
        
        print("=" * 60)
        print(f"Starting Predictors Pipeline - {datetime.now()}")
        print("=" * 60)
        
        # Step 1: Discover new task
        print("\n1. Discovering new prediction task...")
        discovery = TaskDiscovery(
            api_key=self.config.get("openai_api_key"),
            model=self.config.get("model", "gpt-4o")
        )
        
        task = discovery.get_creative_task()
        print(f"   Task: {task['name']}")
        print(f"   Description: {task['description']}")
        
        # Step 2: Generate experiment
        print("\n2. Generating experiment...")
        generator = ExperimentGenerator()
        experiment_path = generator.generate_experiment(
            task,
            api_key=self.config.get("openai_api_key")
        )
        print(f"   Experiment created: {experiment_path}")
        
        # Step 3: Run experiment
        if force_local or self.config.get("local_only", False):
            print("\n3. Running experiment locally...")
            self._run_local(experiment_path)
        elif self.config.get("run_on_vast", True):
            print("\n3. Running experiment on VAST AI...")
            self._run_on_vast(experiment_path)
        else:
            print("\n3. Skipping execution (configured to not run)")
        
        # Step 4: Update history
        print("\n4. Updating history...")
        task["experiment_path"] = experiment_path
        task["completed_at"] = datetime.now().isoformat()
        discovery.record_task(task)
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        
        return task
    
    def _run_local(self, experiment_path: str):
        """Run experiment locally"""
        import subprocess
        
        print(f"   Running training script locally...")
        
        # Change to experiment directory and run
        result = subprocess.run(
            [sys.executable, "train.py"],
            cwd=experiment_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("   Training completed successfully")
        else:
            print(f"   Training failed: {result.stderr}")
    
    def _run_on_vast(self, experiment_path: str):
        """Run experiment on VAST AI"""
        
        runner = VastRunner(api_key=self.config.get("vast_api_key"))
        
        # Estimate cost
        estimated_cost = runner.estimate_cost(
            experiment_path,
            hours=self.config.get("max_training_hours", 3)
        )
        
        if estimated_cost > 0:
            print(f"   Estimated cost: ${estimated_cost:.2f}")
            
            # Run experiment
            results = runner.run_experiment(
                experiment_path,
                min_gpu_ram=self.config.get("min_gpu_ram", 16),
                max_price=self.config.get("max_price_per_hour", 1.0)
            )
            
            print(f"   Experiment completed: {results}")
        else:
            print("   No suitable instances available, skipping")
    
    def weekly_run(self):
        """Entry point for weekly automated runs"""
        try:
            self.run_pipeline()
        except Exception as e:
            print(f"Pipeline failed: {e}")
            
            # Log error
            error_log = {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "type": type(e).__name__
            }
            
            with open("errors.log", 'a') as f:
                f.write(json.dumps(error_log) + "\n")
    
    def list_experiments(self):
        """List all experiments"""
        
        experiments_dir = Path("experiments")
        if not experiments_dir.exists():
            print("No experiments found")
            return
        
        experiments = sorted(experiments_dir.iterdir(), key=lambda x: x.name)
        
        print(f"\nFound {len(experiments)} experiments:\n")
        
        for exp_path in experiments:
            task_file = exp_path / "task.json"
            results_file = exp_path / "results.json"
            
            if task_file.exists():
                with open(task_file, 'r') as f:
                    task = json.load(f)
                
                status = "✓ Completed" if results_file.exists() else "○ Pending"
                print(f"{status} {exp_path.name}")
                print(f"     Task: {task['name']}")
                print(f"     Type: {task['task_type']}")
                
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    print(f"     Loss: {results.get('final_val_loss', 'N/A')}")
                print()


def main():
    parser = argparse.ArgumentParser(description="Predictors Framework Orchestrator")
    parser.add_argument("--run", action="store_true", help="Run a new experiment")
    parser.add_argument("--local", action="store_true", help="Run locally instead of VAST")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    orchestrator = Orchestrator(config_path=args.config)
    
    if args.run:
        orchestrator.run_pipeline(force_local=args.local)
    elif args.list:
        orchestrator.list_experiments()
    else:
        # Default: show status
        print("Predictors Framework")
        print("=" * 40)
        print("\nUsage:")
        print("  python orchestrator.py --run        # Run new experiment")
        print("  python orchestrator.py --run --local # Run locally")
        print("  python orchestrator.py --list       # List experiments")
        print("\nFor weekly automation, add to crontab:")
        print("  0 0 * * 0 cd /path/to/predictors && python orchestrator.py --run")
        
        orchestrator.list_experiments()


if __name__ == "__main__":
    main()