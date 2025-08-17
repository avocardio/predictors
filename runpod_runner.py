#!/usr/bin/env python3
"""
RunPod GPU server management for running experiments
"""
import os
import time
import json
import requests
from pathlib import Path
from typing import Dict, Optional, List
import subprocess


class RunPodRunner:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError("RunPod API key required. Set RUNPOD_API_KEY environment variable.")
        
        self.base_url = "https://api.runpod.io/v2"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def create_pod(self, gpu_type: str = "RTX 4090", min_download: int = 500) -> str:
        """Create a RunPod instance with specified GPU and startup script"""
        
        # Upload startup script and framework
        startup_script = """#!/bin/bash
cd /workspace
git clone https://github.com/avocardio/predictors.git
cd predictors
python startup_script.py
"""
        
        # Pod configuration for ML workload
        config = {
            "name": f"predictors-{int(time.time())}",
            "imageName": "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-runtime",
            "gpuTypeId": gpu_type,
            "cloudType": "SECURE",  # or "COMMUNITY" for cheaper
            "minDownload": min_download,  # Mbps
            "minUpload": 100,
            "diskSizeInGb": 50,
            "containerDiskInGb": 50,
            "volumeInGb": 0,
            "ports": "8888/http,22/tcp",
            "env": [
                {"key": "PYTHONUNBUFFERED", "value": "1"},
                {"key": "WORKSPACE", "value": "/workspace"},
                {"key": "ANTHROPIC_API_KEY", "value": os.getenv("ANTHROPIC_API_KEY", "")}
            ],
            "dockerArgs": "",
            "startScript": startup_script,
            "supportPublicIp": True
        }
        
        response = requests.post(
            f"{self.base_url}/pod/run",
            headers=self.headers,
            json=config
        )
        
        if response.status_code == 200:
            pod_data = response.json()
            pod_id = pod_data["id"]
            print(f"Created pod: {pod_id}")
            
            # Wait for pod to be ready
            self._wait_for_pod(pod_id)
            return pod_id
        else:
            raise RuntimeError(f"Failed to create pod: {response.text}")
    
    def _wait_for_pod(self, pod_id: str, timeout: int = 300):
        """Wait for pod to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{self.base_url}/pod/{pod_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                pod = response.json()
                if pod.get("runtime") and pod["runtime"].get("status") == "RUNNING":
                    print(f"Pod {pod_id} is ready")
                    return pod
            
            time.sleep(5)
        
        raise TimeoutError(f"Pod {pod_id} not ready after {timeout} seconds")
    
    def upload_experiment(self, pod_id: str, experiment_path: str):
        """Upload experiment to pod"""
        
        # Get pod details
        response = requests.get(
            f"{self.base_url}/pod/{pod_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            pod = response.json()
            
            # Get SSH details
            ssh_host = pod["runtime"]["publicIp"]
            ssh_port = 22
            
            print(f"Uploading to {ssh_host}:{ssh_port}")
            
            # Create tar archive
            import tarfile
            tar_path = "/tmp/experiment.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(experiment_path, arcname="experiment")
            
            # Upload via SCP (requires SSH key setup) or use RunPod's file transfer
            # For now, we'll use a simpler approach with base64 encoding
            with open(tar_path, 'rb') as f:
                import base64
                encoded = base64.b64encode(f.read()).decode()
            
            # Execute command to decode and extract on pod
            command = f"""
            cd /workspace && \
            echo '{encoded}' | base64 -d > experiment.tar.gz && \
            tar -xzf experiment.tar.gz && \
            rm experiment.tar.gz
            """
            
            self.execute_command(pod_id, command)
            os.remove(tar_path)
            
            print("Experiment uploaded successfully")
    
    def execute_command(self, pod_id: str, command: str) -> str:
        """Execute command on pod"""
        
        response = requests.post(
            f"{self.base_url}/pod/{pod_id}/exec",
            headers=self.headers,
            json={"command": command}
        )
        
        if response.status_code == 200:
            return response.json().get("output", "")
        else:
            raise RuntimeError(f"Command execution failed: {response.text}")
    
    def run_training(self, pod_id: str) -> Dict:
        """Run training on pod"""
        
        print(f"Starting training on pod {pod_id}")
        
        # Install requirements and run training
        commands = [
            "cd /workspace/experiment && pip install -r /workspace/requirements.txt",
            "cd /workspace/experiment && python train.py"
        ]
        
        for cmd in commands:
            output = self.execute_command(pod_id, cmd)
            print(f"Command output: {output[:500]}...")  # Print first 500 chars
        
        # Monitor training
        return self._monitor_training(pod_id)
    
    def _monitor_training(self, pod_id: str) -> Dict:
        """Monitor training progress"""
        
        print(f"Monitoring training on pod {pod_id}")
        
        while True:
            # Check if results file exists
            check_cmd = "test -f /workspace/experiment/results.json && cat /workspace/experiment/results.json || echo 'PENDING'"
            output = self.execute_command(pod_id, check_cmd)
            
            if output != "PENDING":
                try:
                    results = json.loads(output)
                    print("Training completed!")
                    return results
                except json.JSONDecodeError:
                    pass
            
            # Check logs for errors
            log_cmd = "tail -n 50 /workspace/experiment/train.log 2>/dev/null || echo ''"
            logs = self.execute_command(pod_id, log_cmd)
            
            if "Error" in logs or "Exception" in logs:
                print(f"Training failed. Logs:\n{logs}")
                break
            
            time.sleep(30)
        
        return {"status": "failed"}
    
    def download_results(self, pod_id: str, experiment_path: str):
        """Download results from pod"""
        
        print(f"Downloading results from pod {pod_id}")
        
        # Get results.json
        results_cmd = "cat /workspace/experiment/results.json 2>/dev/null"
        results = self.execute_command(pod_id, results_cmd)
        
        if results:
            results_path = Path(experiment_path) / "results.json"
            with open(results_path, 'w') as f:
                f.write(results)
            print(f"Saved results to {results_path}")
        
        # Get best checkpoint info
        checkpoint_cmd = "ls -la /workspace/experiment/checkpoints/ 2>/dev/null | head -5"
        checkpoints = self.execute_command(pod_id, checkpoint_cmd)
        print(f"Checkpoints:\n{checkpoints}")
    
    def terminate_pod(self, pod_id: str):
        """Terminate pod to stop billing"""
        
        response = requests.delete(
            f"{self.base_url}/pod/{pod_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            print(f"Pod {pod_id} terminated")
        else:
            print(f"Warning: Failed to terminate pod {pod_id}: {response.text}")
    
    def estimate_cost(self, gpu_type: str = "RTX 4090", hours: float = 2.0) -> float:
        """Estimate cost for running experiment"""
        
        # RunPod pricing (approximate, as of 2025)
        gpu_prices = {
            "RTX 4090": 0.44,  # $/hr
            "RTX A6000": 0.79,
            "A100 40GB": 1.09,
            "A100 80GB": 1.89,
            "H100 80GB": 2.99
        }
        
        price_per_hour = gpu_prices.get(gpu_type, 1.0)
        cost = price_per_hour * hours
        
        print(f"Estimated cost: ${cost:.2f} for {hours} hours on {gpu_type}")
        return cost