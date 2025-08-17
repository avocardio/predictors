import os
import requests
import kaggle
from datasets import load_dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import zipfile
import tarfile
from pathlib import Path


class DataAcquisition:
    def __init__(self):
        self.kaggle_configured = self._check_kaggle_auth()
        
    def _check_kaggle_auth(self) -> bool:
        try:
            kaggle.api.authenticate()
            return True
        except Exception as e:
            print(f"Kaggle auth not configured: {e}")
            return False
    
    def download_data(self, source: Dict, target_dir: str) -> str:
        """Download data from various sources"""
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        source_type = source['type'].lower()
        
        if source_type == 'kaggle':
            return self._download_kaggle(source['identifier'], target_dir)
        elif source_type == 'huggingface':
            return self._download_huggingface(source['identifier'], target_dir)
        elif source_type == 'openml':
            return self._download_openml(source['identifier'], target_dir)
        elif source_type == 'url':
            return self._download_url(source['identifier'], target_dir)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    
    def _download_kaggle(self, identifier: str, target_dir: str) -> str:
        if not self.kaggle_configured:
            raise RuntimeError("Kaggle API not configured. Set up ~/.kaggle/kaggle.json")
        
        # identifier format: "username/dataset-name" or competition name
        if '/' in identifier:  # It's a dataset
            kaggle.api.dataset_download_files(identifier, path=target_dir, unzip=True)
        else:  # It's a competition
            kaggle.api.competition_download_files(identifier, path=target_dir, quiet=False)
            # Unzip if needed
            for file in Path(target_dir).glob("*.zip"):
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                file.unlink()
        
        return target_dir
    
    def _download_huggingface(self, identifier: str, target_dir: str) -> str:
        # Load dataset from HuggingFace
        dataset = load_dataset(identifier)
        
        # Save to disk
        save_path = Path(target_dir) / "data"
        dataset.save_to_disk(str(save_path))
        
        # Also save as CSV for easier inspection
        for split_name, split_data in dataset.items():
            df = split_data.to_pandas()
            df.to_csv(Path(target_dir) / f"{split_name}.csv", index=False)
        
        return str(save_path)
    
    def _download_openml(self, identifier: str, target_dir: str) -> str:
        # OpenML dataset ID
        import openml
        
        dataset = openml.datasets.get_dataset(identifier)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        
        # Save as CSV
        data = pd.concat([X, y], axis=1)
        data.to_csv(Path(target_dir) / "data.csv", index=False)
        
        # Save metadata
        metadata = {
            "name": dataset.name,
            "features": attribute_names,
            "categorical": categorical_indicator,
            "target": dataset.default_target_attribute
        }
        
        import json
        with open(Path(target_dir) / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return target_dir
    
    def _download_url(self, url: str, target_dir: str) -> str:
        # Download file from URL
        filename = url.split('/')[-1]
        if not filename:
            filename = "data.bin"
        
        filepath = Path(target_dir) / filename
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract if compressed
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            filepath.unlink()
        elif filename.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(filepath, 'r:gz') as tar_ref:
                tar_ref.extractall(target_dir)
            filepath.unlink()
        elif filename.endswith('.tar'):
            with tarfile.open(filepath, 'r') as tar_ref:
                tar_ref.extractall(target_dir)
            filepath.unlink()
        
        return target_dir
    
    def download_multiple(self, sources: List[Dict], base_dir: str) -> Dict[str, str]:
        """Download multiple data sources"""
        results = {}
        
        for i, source in enumerate(sources):
            target_dir = Path(base_dir) / f"source_{i}"
            try:
                path = self.download_data(source, str(target_dir))
                results[source.get('description', f'source_{i}')] = path
                print(f"Downloaded {source['identifier']} to {path}")
            except Exception as e:
                print(f"Failed to download {source['identifier']}: {e}")
                results[source.get('description', f'source_{i}')] = None
        
        return results
    
    def validate_data(self, data_dir: str) -> Dict[str, Any]:
        """Basic validation of downloaded data"""
        path = Path(data_dir)
        
        info = {
            "exists": path.exists(),
            "files": [],
            "total_size_mb": 0
        }
        
        if path.exists():
            for file in path.rglob('*'):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    info["files"].append({
                        "name": str(file.relative_to(path)),
                        "size_mb": round(size_mb, 2)
                    })
                    info["total_size_mb"] += size_mb
        
        info["total_size_mb"] = round(info["total_size_mb"], 2)
        return info