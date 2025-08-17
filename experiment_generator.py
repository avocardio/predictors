import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from task_discovery import TaskDiscovery
from data_acquisition import DataAcquisition


class ExperimentGenerator:
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def generate_experiment(self, task: Dict, api_key: Optional[str] = None) -> str:
        """Generate a complete experiment folder for a task"""
        
        # Create experiment folder
        exp_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task['name']}"
        exp_path = self.base_dir / exp_name
        exp_path.mkdir(parents=True, exist_ok=True)
        
        # Save task definition
        with open(exp_path / "task.json", 'w') as f:
            json.dump(task, f, indent=2)
        
        # Copy base model
        shutil.copy("base_model.py", exp_path / "model.py")
        
        # Generate dataloader using reasoning model
        discovery = TaskDiscovery(api_key=api_key)
        dataloader_code = discovery.generate_dataloader_code(task)
        with open(exp_path / "dataloader.py", 'w') as f:
            f.write(dataloader_code)
        
        # Get model dimensions
        dimensions = discovery.analyze_task_dimensions(task)
        
        # Generate training script
        train_script = self._generate_train_script(task, dimensions)
        with open(exp_path / "train.py", 'w') as f:
            f.write(train_script)
        
        # Download data
        data_dir = exp_path / "data"
        acquisition = DataAcquisition()
        data_paths = acquisition.download_multiple(task['data_sources'], str(data_dir))
        
        # Save data paths
        with open(exp_path / "data_paths.json", 'w') as f:
            json.dump(data_paths, f, indent=2)
        
        # Validate data
        validation_info = {}
        for name, path in data_paths.items():
            if path:
                validation_info[name] = acquisition.validate_data(path)
        
        with open(exp_path / "data_validation.json", 'w') as f:
            json.dump(validation_info, f, indent=2)
        
        print(f"Generated experiment: {exp_path}")
        return str(exp_path)
    
    def _generate_train_script(self, task: Dict, dimensions: Dict) -> str:
        return f"""import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import BaseTransformer
from dataloader import get_dataloaders

def main():
    # Task: {task['name']}
    # Type: {task['task_type']}
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32,
        num_workers=4
    )
    
    # Initialize model
    model = BaseTransformer(
        input_dim={dimensions['input_dim']},
        output_dim={dimensions['output_dim']},
        d_model={dimensions['suggested_d_model']},
        nhead=8,
        num_layers={dimensions['suggested_layers']},
        dim_feedforward={dimensions['suggested_d_model'] * 4},
        dropout=0.1,
        learning_rate=1e-4,
        warmup_steps=1000,
        max_steps={dimensions['estimated_training_steps']},
        task_type='{task['task_type']}'
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='best-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Logger
    logger = CSVLogger('logs', name='training')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        val_check_interval=0.25
    )
    
    # Train
    print(f"Starting training for task: {task['name']}")
    print(f"Evaluation metric: {task['evaluation_metric']}")
    print(f"Expected baseline: {task['expected_baseline']}")
    
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    print("\\nTesting best model...")
    trainer.test(model, test_loader, ckpt_path='best')
    
    # Save final results
    import json
    results = {{
        'task': '{task['name']}',
        'final_val_loss': float(trainer.callback_metrics.get('val_loss', -1)),
        'epochs_trained': trainer.current_epoch,
        'best_checkpoint': checkpoint_callback.best_model_path
    }}
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\\nTraining complete!")
    print(f"Results saved to results.json")

if __name__ == '__main__':
    main()
"""