import subprocess
import json
import time
import os
from pathlib import Path
from typing import Dict, Optional
import requests


class VastRunner:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("VAST_API_KEY")
        if not self.api_key:
            raise ValueError("VAST API key required. Set VAST_API_KEY environment variable.")
        
        self.base_url = "https://console.vast.ai/api/v0"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    def find_instance(self, min_gpu_ram: int = 16, max_price: float = 1.0) -> Optional[Dict]:
        """Find suitable GPU instance"""
        
        # Search for available instances
        params = {
            "q": json.dumps({
                "verified": {"eq": True},
                "external": {"eq": False},
                "rentable": {"eq": True},
                "gpu_ram": {"gte": min_gpu_ram * 1000},  # Convert GB to MB
                "dph_total": {"lte": max_price},
                "cuda_vers": {"gte": 11.0},
                "inet_up": {"gte": 100},
                "inet_down": {"gte": 100}
            }),
            "order": "dph_total",
            "limit": 5
        }
        
        response = requests.get(
            f"{self.base_url}/bundles",
            headers=self.headers,
            params=params
        )
        
        if response.status_code == 200:
            offers = response.json().get("offers", [])
            if offers:
                return offers[0]  # Return cheapest suitable option
        
        return None
    
    def create_instance(self, offer_id: int, experiment_path: str) -> str:
        """Create and configure instance"""
        
        # Prepare startup script
        startup_script = f"""#!/bin/bash
apt-get update
apt-get install -y python3-pip git
pip install -r /workspace/requirements.txt
cd /workspace
python train.py
"""
        
        # Create instance
        payload = {
            "client_id": "me",
            "offer_id": offer_id,
            "image": "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
            "env": {"PYTHONUNBUFFERED": "1"},
            "onstart": startup_script,
            "disk": 50  # GB
        }
        
        response = requests.post(
            f"{self.base_url}/instances",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            instance = response.json()
            instance_id = instance["id"]
            
            # Wait for instance to be ready
            self._wait_for_instance(instance_id)
            
            # Upload experiment files
            self._upload_files(instance_id, experiment_path)
            
            return instance_id
        else:
            raise RuntimeError(f"Failed to create instance: {response.text}")
    
    def _wait_for_instance(self, instance_id: str, timeout: int = 300):
        """Wait for instance to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{self.base_url}/instances/{instance_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                instance = response.json()
                if instance["status"] == "running":
                    print(f"Instance {instance_id} is ready")
                    return
            
            time.sleep(10)
        
        raise TimeoutError(f"Instance {instance_id} not ready after {timeout} seconds")
    
    def _upload_files(self, instance_id: str, experiment_path: str):
        """Upload experiment files to instance"""
        
        # Get instance SSH details
        response = requests.get(
            f"{self.base_url}/instances/{instance_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            instance = response.json()
            ssh_host = instance["ssh_host"]
            ssh_port = instance["ssh_port"]
            
            # Use SCP to upload files (requires SSH key setup)
            # For simplicity, we'll use vast.ai's file upload API
            print(f"Uploading files to instance {instance_id}")
            
            # Package experiment as tar
            import tarfile
            tar_path = "/tmp/experiment.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(experiment_path, arcname="workspace")
            
            # Upload via API
            with open(tar_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{self.base_url}/instances/{instance_id}/upload",
                    headers=self.headers,
                    files=files
                )
            
            os.remove(tar_path)
            
            if response.status_code == 200:
                print("Files uploaded successfully")
            else:
                raise RuntimeError(f"Failed to upload files: {response.text}")
    
    def run_experiment(self, experiment_path: str, min_gpu_ram: int = 16, max_price: float = 1.0) -> Dict:
        """Full pipeline to run experiment on VAST"""
        
        print(f"Finding suitable instance...")
        offer = self.find_instance(min_gpu_ram, max_price)
        
        if not offer:
            raise RuntimeError("No suitable instances available")
        
        print(f"Found instance: {offer['gpu_name']} at ${offer['dph_total']}/hour")
        
        instance_id = self.create_instance(offer["id"], experiment_path)
        print(f"Created instance: {instance_id}")
        
        # Monitor training
        results = self._monitor_training(instance_id)
        
        # Download results
        self._download_results(instance_id, experiment_path)
        
        # Destroy instance
        self.destroy_instance(instance_id)
        
        return results
    
    def _monitor_training(self, instance_id: str, check_interval: int = 60) -> Dict:
        """Monitor training progress"""
        
        print(f"Monitoring training on instance {instance_id}")
        
        while True:
            # Check logs
            response = requests.get(
                f"{self.base_url}/instances/{instance_id}/logs",
                headers=self.headers
            )
            
            if response.status_code == 200:
                logs = response.json().get("logs", "")
                
                # Check if training completed
                if "Training complete!" in logs:
                    print("Training completed successfully")
                    break
                elif "Error" in logs or "Exception" in logs:
                    print(f"Training failed. Check logs.")
                    break
            
            time.sleep(check_interval)
        
        return {"status": "completed", "instance_id": instance_id}
    
    def _download_results(self, instance_id: str, experiment_path: str):
        """Download results from instance"""
        
        print(f"Downloading results from instance {instance_id}")
        
        # Download results.json and checkpoints
        files_to_download = ["results.json", "checkpoints/", "logs/"]
        
        for file_path in files_to_download:
            response = requests.get(
                f"{self.base_url}/instances/{instance_id}/download",
                headers=self.headers,
                params={"path": f"/workspace/{file_path}"}
            )
            
            if response.status_code == 200:
                # Save to experiment folder
                local_path = Path(experiment_path) / file_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"Downloaded {file_path}")
    
    def destroy_instance(self, instance_id: str):
        """Destroy instance to stop billing"""
        
        response = requests.delete(
            f"{self.base_url}/instances/{instance_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            print(f"Instance {instance_id} destroyed")
        else:
            print(f"Warning: Failed to destroy instance {instance_id}")
    
    def estimate_cost(self, experiment_path: str, hours: float = 2.0) -> float:
        """Estimate cost for running experiment"""
        
        # Find cheapest suitable instance
        offer = self.find_instance()
        
        if offer:
            cost = offer["dph_total"] * hours
            print(f"Estimated cost: ${cost:.2f} for {hours} hours on {offer['gpu_name']}")
            return cost
        else:
            print("No suitable instances available")
            return 0.0