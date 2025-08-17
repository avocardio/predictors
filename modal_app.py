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
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0", 
        "datasets>=2.14.0",
        "kaggle>=1.5.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0"
    ])
)

# Create persistent volume for history and results
volume = modal.Volume.from_name("predictors-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",  # Use A100 for cost efficiency
    volumes={"/data": volume},
    timeout=10800,  # 3 hours max
    secrets=[
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("kaggle-credentials", required=False)
    ],
    schedule=modal.Cron("0 0 * * 0")  # Every Sunday at midnight UTC
)
def weekly_prediction_discovery():
    """
    Main function that runs weekly to discover and train a new prediction task
    """
    
    print("Starting Weekly Discovery")
    print(f"Time: {datetime.now()}")
    print(f"GPU: {modal.gpu.count()} available")
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
        results = run_claude_pipeline(workspace)
        
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


def run_claude_pipeline(workspace: Path) -> dict:
    """Run the Claude Code pipeline for task discovery and training"""
    
    print("\n🤖 Starting Claude Code Pipeline")
    
    # Create pipeline instructions
    pipeline_instructions = """
You are running the Predictors framework for automated ML task discovery.

EXECUTE THIS COMPLETE PIPELINE:

1. **DISCOVER NOVEL TASK**
   - Read history.json to see previous tasks
   - Think creatively about a NEW cross-domain prediction task
   - Focus on interesting correlations (weather ↔ social media, music ↔ markets, etc.)
   - Create experiments/next_task.json with task specification

2. **GENERATE EXPERIMENT**
   - Create timestamped experiment folder in experiments/
   - Copy base_model.py to experiment folder as model.py
   - Generate custom dataloader.py for the specific task
   - Generate configured train.py script
   - Download data using APIs (HuggingFace, Kaggle, etc.)

3. **RUN TRAINING**
   - Execute python train.py in the experiment folder
   - Monitor training progress
   - Ensure results.json is created

4. **UPDATE HISTORY**
   - Add completed task to history.json
   - Include results and timestamp

5. **SIGNAL COMPLETION**
   - Create file called "PIPELINE_COMPLETE.txt" when done
   - Include summary of what was accomplished

Requirements:
- Use API-accessible data only
- Keep data size < 1GB
- Focus on transformer-solvable tasks
- Be genuinely creative and novel

START THE PIPELINE NOW.
    """
    
    # Write instructions to file
    instructions_file = workspace / "pipeline_instructions.txt"
    instructions_file.write_text(pipeline_instructions)
    
    print("📝 Pipeline instructions created")
    
    # Execute Claude Code with the instructions
    print("🧠 Executing Claude Code...")
    
    try:
        # Set environment variables from Modal secrets
        env = os.environ.copy()
        env['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')
        if os.getenv('KAGGLE_USERNAME'):
            env['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
        if os.getenv('KAGGLE_KEY'):
            env['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
        
        # First-time Claude Code authentication (if needed)
        print("🔐 Checking Claude Code authentication...")
        auth_check = subprocess.run(
            'echo "/login" | claude --print',
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        if auth_check.returncode != 0:
            print("⚠️ Claude Code authentication may be required - proceeding anyway...")
        
        # Execute main pipeline
        result = subprocess.run(
            'echo "$(cat pipeline_instructions.txt)" | claude --dangerously-skip-permissions --print --max-turns 25',
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=12000,  # 3+ hours for full pipeline
            env=env
        )
        
        print(f"🔄 Claude Code execution completed (return code: {result.returncode})")
        
        if result.stdout:
            print("📤 Claude Output (last 1000 chars):")
            print(result.stdout[-1000:])
        
        if result.stderr:
            print("⚠️ Claude Errors:")
            print(result.stderr[-500:])
        
        # Check if pipeline completed
        completion_file = workspace / "PIPELINE_COMPLETE.txt"
        if completion_file.exists():
            print("✅ Pipeline completed successfully!")
            
            # Gather results
            results = gather_pipeline_results(workspace)
            return results
        else:
            print("⚠️ Pipeline may not have completed fully")
            return {"status": "incomplete", "stdout": result.stdout[-1000:]}
    
    except subprocess.TimeoutExpired:
        print("⏰ Claude Code timed out after 3+ hours")
        return {"status": "timeout"}
    
    except Exception as e:
        print(f"💥 Claude Code execution failed: {e}")
        return {"status": "error", "error": str(e)}


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
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("kaggle-credentials", required=False)
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