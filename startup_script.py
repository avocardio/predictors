#!/usr/bin/env python3
"""
Startup script for RunPod GPU instances
Installs dependencies and launches Claude Code
"""
import subprocess
import sys
import os
from pathlib import Path


def setup_environment():
    """Setup the GPU server environment"""
    
    print("Setting up Predictors framework on GPU server...")
    
    # Update system
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "nodejs", "npm", "git"], check=True)
    
    # Install Claude Code
    print("Installing Claude Code...")
    subprocess.run(["npm", "install", "-g", "@anthropic-ai/claude-code"], check=True)
    
    # Install Python dependencies
    print("Installing Python dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pip"], check=True)
    
    # Install PyTorch with GPU support
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", "--index-url", 
        "https://download.pytorch.org/whl/cu118"
    ], check=True)
    
    # Install other requirements
    requirements = [
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
    ]
    
    for req in requirements:
        subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)
    
    print("Environment setup complete!")


def create_config():
    """Create initial config and history files"""
    
    workspace = Path("/workspace/predictors")
    workspace.mkdir(exist_ok=True)
    
    # Create empty history if it doesn't exist
    history_file = workspace / "history.json"
    if not history_file.exists():
        import json
        with open(history_file, 'w') as f:
            json.dump({"tasks": []}, f, indent=2)
    
    # Create experiments directory
    (workspace / "experiments").mkdir(exist_ok=True)
    
    print(f"Workspace initialized at {workspace}")


def launch_claude():
    """Launch Claude Code with the right permissions"""
    
    print("\n" + "="*60)
    print("LAUNCHING CLAUDE CODE")
    print("="*60)
    print("\nAvailable commands:")
    print("  /discover  - Discover a new prediction task")
    print("  /generate  - Generate experiment for a task")
    print("  /cleanup   - Clean up an experiment folder")
    print("  /run       - Run complete pipeline")
    print("\nStarting Claude Code with dangerous permissions...")
    print("Remember: This Claude instance can modify files and run commands!")
    print("="*60)
    
    # Change to workspace
    os.chdir("/workspace/predictors")
    
    # Launch Claude Code
    subprocess.run([
        "claude", 
        "--dangerously-skip-permissions"
    ])


def main():
    """Main startup sequence"""
    
    print("Predictors Framework - GPU Server Startup")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Create config
    create_config()
    
    # Launch Claude Code
    launch_claude()


if __name__ == "__main__":
    main()