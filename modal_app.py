#!/usr/bin/env python3
"""
Modal serverless app for automated weekly prediction task discovery
"""
import modal
import subprocess
import json
import os
from datetime import datetime
from pathlib import Path
import tempfile
import shutil


# Create Modal app
app = modal.App("predictors-framework")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("nodejs", "npm", "git", "curl")
    .run_commands("npm install -g @anthropic-ai/claude-code")
    .pip_install([
        "torch>=2.0.0",
        "torchvision", 
        "torchaudio",
        "lightning>=2.0.0",
        "transformers>=4.30.0", 
        "datasets>=2.14.0",
        "kaggle>=1.5.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "anthropic>=0.40.0",
        "output-shape",
        "wandb>=0.16.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ])
    .add_local_file("task_discovery.py", remote_path="/workspace/task_discovery.py")
    .add_local_file("experiment_generator.py", remote_path="/workspace/experiment_generator.py")
    .add_local_file("reasoning_agent.py", remote_path="/workspace/reasoning_agent.py")
    .add_local_file("base_model.py", remote_path="/workspace/base_model.py")
    .add_local_file("data_acquisition.py", remote_path="/workspace/data_acquisition.py")
    .add_local_file("CLAUDE.md", remote_path="/workspace/CLAUDE.md")
)

# Create persistent volume for history and results
volume = modal.Volume.from_name("predictors-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",  # Use A100 for cost efficiency
    volumes={"/data": volume},
    timeout=10800,  # 3 hours max
    secrets=[
        modal.Secret.from_name("anthropic-secret"),
        modal.Secret.from_name("kaggle"),
        modal.Secret.from_name("wandb")
    ],
    schedule=modal.Cron("0 0 * * 0")  # Every Sunday at midnight UTC
)
def weekly_prediction_discovery():
    """
    Main function that runs weekly to discover and train a new prediction task
    """
    
    print("🚀 Starting Weekly Prediction Discovery")
    print(f"Time: {datetime.now()}")
    print(f"GPU: A100 allocated")
    print("="*60)
    
    # Setup workspace
    workspace = Path("/tmp/predictors")
    workspace.mkdir(exist_ok=True)
    os.chdir(workspace)
    
    # Clone the repository
    subprocess.run([
        "git", "clone", 
        "https://github.com/avocardio/predictors.git", 
        "."
    ], check=True)
    
    # Load persistent history
    history_file = Path("/data/history.json")
    local_history = workspace / "history.json"
    
    if history_file.exists():
        shutil.copy(history_file, local_history)
        print(f"📚 Loaded history with {len(json.loads(local_history.read_text()).get('tasks', []))} previous tasks")
    else:
        local_history.write_text('{"tasks": []}')
        print("📚 Created new history file")
    
    # Run the automated pipeline
    try:
        results = run_api_pipeline(workspace)
        
        # Save results to persistent storage
        save_results_to_volume(workspace, results)
        
        print("✅ Weekly discovery completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        
        # Save error info
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "failed"
        }
        
        error_file = Path("/data") / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        error_file.write_text(json.dumps(error_info, indent=2))
        
        raise


def run_api_pipeline(workspace: Path) -> dict:
    """Run the reasoning-based pipeline for task discovery and training"""
    
    print("\n🤖 Starting Reasoning Pipeline")
    
    try:
        # Import our modules from workspace
        import sys
        sys.path.insert(0, "/workspace")
        from reasoning_agent import ReasoningAgent
        from task_discovery import TaskDiscovery
        from experiment_generator import ExperimentGenerator
        
        # Initialize with API key
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        # Load history
        history_file = workspace / "history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f).get("tasks", [])
        else:
            history = []
        
        # Use reasoning agent for multi-step reasoning
        agent = ReasoningAgent(api_key)
        task = agent.run_full_pipeline(history)
        
        # Save reasoning log
        reasoning_log = workspace / f"reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        agent.save_reasoning_log(reasoning_log)
        
        # Generate experiment with the reasoned task and history
        generator = ExperimentGenerator()
        experiment_path = generator.generate_experiment(task, api_key, history)
        print(f"   Experiment created: {experiment_path}")
        
        # Run training with error recovery
        print("🚀 Starting training...")
        training_result = run_training_with_recovery(experiment_path, task, history, api_key)
        
        # Update history
        print("📝 Updating history...")
        discovery = TaskDiscovery(api_key)
        task["experiment_path"] = experiment_path
        task["training_result"] = training_result
        task["completed_at"] = datetime.now().isoformat()
        discovery.save_task(task)
        
        # Check training status for accurate final message
        if training_result.get("status") == "completed":
            print("✅ Pipeline completed successfully with training!")
            status = "fully_completed"
        elif training_result.get("status") == "failed":
            print("⚠️ Pipeline completed but training failed")
            status = "completed_training_failed" 
        else:
            print("⚠️ Pipeline completed but training status unclear")
            status = "completed_unclear"
        
        return {"status": status, "task": task, "experiment_path": experiment_path, "training_result": training_result, "reasoning_traces": agent.reasoning_traces}
        
    except Exception as e:
        print(f"💥 Pipeline failed: {e}")
        return {"status": "error", "error": str(e)}


def run_training_with_recovery(experiment_path: str, task: dict, history: list, api_key: str, max_attempts: int = 3) -> dict:
    """Run training with automatic error recovery using LLM fixes"""
    
    for attempt in range(max_attempts):
        print(f"🔧 Training attempt {attempt + 1}/{max_attempts}")
        
        # Try to run the experiment
        result = run_training(experiment_path)
        
        if result.get("status") == "completed":
            print(f"✅ Training succeeded on attempt {attempt + 1}")
            return result
        elif result.get("status") in ["completed_no_training", "completed_no_results"]:
            print(f"⚠️ Process ran but no actual training occurred on attempt {attempt + 1}")
        
        if attempt < max_attempts - 1:  # Don't fix on last attempt
            print(f"❌ Attempt {attempt + 1} failed, asking LLM to fix...")
            
            # Get the error details
            error_msg = result.get("error", "")
            stdout_msg = result.get("stdout", "")
            
            # Ask LLM to fix the code
            fixed_code = fix_experiment_code(experiment_path, error_msg, stdout_msg, task, history, api_key)
            
            if fixed_code:
                # Write the fixed code
                with open(Path(experiment_path) / "experiment.py", 'w') as f:
                    f.write(fixed_code)
                print(f"🔧 Code fixed, retrying...")
            else:
                print(f"⚠️ LLM couldn't generate fix, retrying with original code...")
    
    print(f"❌ All {max_attempts} attempts failed")
    return result


def run_training(experiment_path: str) -> dict:
    """Run training in the experiment directory"""
    import sys
    try:
        # Print what we're about to run
        print(f"   📂 Running in: {experiment_path}")
        print(f"   🐍 Command: python experiment.py")
        
        result = subprocess.run(
            [sys.executable, "experiment.py"],
            cwd=experiment_path,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout for training
        )
        
        # Always show some output
        if result.stdout:
            print("   📋 Training output (first 500 chars):")
            print(f"   {result.stdout[:500]}")
        
        if result.stderr:
            print("   ⚠️ Stderr output:")
            print(f"   {result.stderr[:500]}")
        
        if result.returncode == 0:
            # Check if training actually happened
            if "Epoch" in result.stdout or "train_loss" in result.stdout or "Training" in result.stdout:
                print("   ✅ Training completed with epochs")
            else:
                print("   ⚠️ Process completed but no training output detected")
                return {"status": "completed_no_training", "error": "No training output found", "stdout": result.stdout[:1000]}
            
            # Try to read results
            results_file = Path(experiment_path) / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    print(f"   📊 Results found: {results}")
                    return {"status": "completed", "results": results}
            else:
                print("   ⚠️ No results.json found")
                return {"status": "completed_no_results", "note": "Training finished but no results.json found", "stdout": result.stdout[:1000]}
        else:
            print(f"   ❌ Training failed with return code {result.returncode}")
            print(f"   Error: {result.stderr[:500] if result.stderr else 'No stderr'}")
            return {"status": "failed", "error": result.stderr, "stdout": result.stdout}
            
    except subprocess.TimeoutExpired:
        print("   ⏰ Training timed out after 2 hours")
        return {"status": "timeout"}
    except Exception as e:
        print(f"   💥 Training error: {e}")
        return {"status": "error", "error": str(e)}


def fix_experiment_code(experiment_path: str, error_msg: str, stdout_msg: str, task: dict, history: list, api_key: str) -> str:
    """Use LLM to fix the experiment code based on the error"""
    try:
        from reasoning_agent import ReasoningAgent
        
        # Read the current broken code
        current_code = ""
        code_file = Path(experiment_path) / "experiment.py"
        if code_file.exists():
            current_code = code_file.read_text()
        
        # Read the data analysis
        data_analysis = task.get("data_analysis", {})
        
        agent = ReasoningAgent(api_key)
        
        # Create a detailed error fixing prompt
        fix_prompt = f"""FIX THE BROKEN CODE BELOW. Return ONLY the corrected Python code, no explanations.

ORIGINAL TASK: {task['name']} - {task['description']}

ERROR THAT OCCURRED:
{error_msg}

STDOUT OUTPUT:
{stdout_msg[:500]}

PREVIOUS SIMILAR ERRORS FROM HISTORY:
{get_similar_errors(history, error_msg)}

AVAILABLE DATA STRUCTURE: {json.dumps(data_analysis, indent=2)}

BROKEN CODE TO FIX:
```python
{current_code}
```

COMMON FIXES NEEDED:
- Use correct file paths from data analysis
- Handle missing columns gracefully
- Fix data type conversions
- Check file exists before reading
- Handle empty dataframes
- Fix tensor shape mismatches

RETURN ONLY THE FIXED PYTHON CODE:"""

        response = agent._api_call_with_retry(
            model=agent.model,
            max_tokens=4000,
            tools=agent.tools,
            system=[{
                "type": "text", 
                "text": agent.cached_system_prompt,
                "cache_control": {"type": "ephemeral"}
            }],
            messages=[{"role": "user", "content": fix_prompt}]
        )
        
        if hasattr(response, 'usage'):
            agent.total_input_tokens += response.usage.input_tokens
            agent.total_output_tokens += response.usage.output_tokens
        
        fixed_code = response.content[0].text
        return agent._extract_python_code(fixed_code)
        
    except Exception as e:
        print(f"❌ Error in fix_experiment_code: {e}")
        return None


def get_similar_errors(history: list, current_error: str) -> str:
    """Get similar errors from history to help with fixes"""
    similar_errors = []
    current_error_lower = current_error.lower()
    
    for task in history[-10:]:  # Last 10 tasks
        if task.get('training_result', {}).get('status') == 'failed':
            error = task.get('training_result', {}).get('error', '')
            if error:
                # Check for similar error types
                if any(keyword in current_error_lower and keyword in error.lower() 
                      for keyword in ['filenotfound', 'attributeerror', 'keyerror', 'valueerror', 'typeerror']):
                    similar_errors.append(f"- {error[:150]}")
    
    return '\n'.join(similar_errors[:3]) if similar_errors else "No similar errors in recent history"


def gather_pipeline_results(workspace: Path) -> dict:
    """Gather results from the completed pipeline"""
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
        "experiments": [],
        "latest_results": None
    }
    
    # Check experiments directory
    experiments_dir = workspace / "experiments"
    if experiments_dir.exists():
        experiments = list(experiments_dir.iterdir())
        results["experiments"] = [exp.name for exp in experiments if exp.is_dir()]
        
        print(f"📊 Found {len(experiments)} experiments")
        
        # Get latest experiment results
        if experiments:
            latest_exp = max(experiments, key=lambda x: x.name)
            results_file = latest_exp / "results.json"
            
            if results_file.exists():
                try:
                    results["latest_results"] = json.loads(results_file.read_text())
                    print(f"📈 Latest results: {results['latest_results']}")
                except Exception as e:
                    print(f"⚠️ Could not read results: {e}")
    
    # Check if history was updated
    history_file = workspace / "history.json"
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text())
            results["total_tasks"] = len(history.get("tasks", []))
            print(f"📚 Total tasks in history: {results['total_tasks']}")
        except Exception as e:
            print(f"⚠️ Could not read history: {e}")
    
    return results


def save_results_to_volume(workspace: Path, results: dict):
    """Save results and history to persistent volume"""
    
    print("💾 Saving results to persistent storage...")
    
    # Save updated history
    local_history = workspace / "history.json"
    if local_history.exists():
        shutil.copy(local_history, "/data/history.json")
        print("📚 History updated in persistent storage")
    
    # Save run summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path("/data") / f"run_summary_{timestamp}.json"
    summary_file.write_text(json.dumps(results, indent=2))
    print(f"📊 Run summary saved: {summary_file}")
    
    # Save latest experiment if exists
    experiments_dir = workspace / "experiments"
    if experiments_dir.exists():
        experiments = list(experiments_dir.iterdir())
        if experiments:
            latest_exp = max(experiments, key=lambda x: x.name)
            
            # Copy key files from latest experiment
            exp_backup = Path("/data") / f"experiment_{timestamp}"
            exp_backup.mkdir(exist_ok=True)
            
            for file_name in ["task.json", "results.json"]:
                source_file = latest_exp / file_name
                if source_file.exists():
                    shutil.copy(source_file, exp_backup / file_name)
            
            print(f"🗂️ Experiment backup saved: {exp_backup}")


# Manual trigger function for testing
@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": volume},
    timeout=7200,  # 2 hours for testing
    secrets=[
        modal.Secret.from_name("anthropic-secret"),
        modal.Secret.from_name("kaggle"),
        modal.Secret.from_name("wandb")
    ]
)
def manual_run():
    """Manual trigger for testing the pipeline"""
    print("🧪 Manual test run triggered")
    return weekly_prediction_discovery.local()


# Function to check history and results
@app.function(
    image=modal.Image.debian_slim(python_version="3.10").pip_install("modal"),
    volumes={"/data": volume}
)
def check_status():
    """Check the status of previous runs"""
    
    data_dir = Path("/data")
    
    # List all files
    if data_dir.exists():
        files = list(data_dir.glob("*"))
        print(f"📁 Files in persistent storage: {len(files)}")
        
        for file in sorted(files)[-10:]:  # Last 10 files
            print(f"  {file.name}")
        
        # Show history summary
        history_file = data_dir / "history.json"
        if history_file.exists():
            history = json.loads(history_file.read_text())
            tasks = history.get("tasks", [])
            print(f"\n📚 Total tasks discovered: {len(tasks)}")
            
            for task in tasks[-3:]:  # Last 3 tasks
                print(f"  - {task.get('name', 'Unknown')}: {task.get('created_at', 'No date')}")
    
    return {"status": "checked", "files": len(files) if 'files' in locals() else 0}


if __name__ == "__main__":
    # This allows local testing
    print("Predictors Framework - Modal App")
    print("Deploy with: modal deploy modal_app.py")
    print("Manual run: modal run modal_app.py::manual_run")
    print("Check status: modal run modal_app.py::check_status")