import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from data_acquisition import DataAcquisition


class ExperimentGenerator:
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def generate_experiment(self, task: Dict, api_key: Optional[str] = None, history: List[Dict] = None) -> str:
        """Generate a complete experiment folder for a task"""
        
        # Create experiment folder
        exp_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task['name']}"
        exp_path = self.base_dir / exp_name
        exp_path.mkdir(parents=True, exist_ok=True)
        
        # Save task definition
        with open(exp_path / "task.json", 'w') as f:
            json.dump(task, f, indent=2)
        
        # Download data first
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
        
        # Use reasoning agent to analyze data and generate complete experiment
        if api_key:
            from reasoning_agent import ReasoningAgent
            agent = ReasoningAgent(api_key)
            
            # Analyze the downloaded data
            task = agent.analyze_downloaded_data(task, data_paths)
            
            # Generate complete experiment code with history
            task = agent.generate_code(task, history)
            
            # Write the complete experiment file
            with open(exp_path / "experiment.py", 'w') as f:
                f.write(task["generated_code"])
        else:
            # Fallback: generate basic experiment without real data analysis
            basic_code = self._generate_basic_experiment(task)
            with open(exp_path / "experiment.py", 'w') as f:
                f.write(basic_code)
        
        print(f"Generated experiment: {exp_path}")
        return str(exp_path)
    
    def _generate_basic_experiment(self, task: Dict) -> str:
        """Fallback experiment generation without LLM analysis"""
        return f'''import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Basic fallback experiment for {task['name']}
class BasicDataset(Dataset):
    def __init__(self, split='train'):
        self.data = np.random.randn(100, 32, 10)  # (samples, seq_len, features)
        self.labels = np.random.randn(100, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {{
            'input': torch.tensor(self.data[idx], dtype=torch.float32),
            'target': torch.tensor(self.labels[idx], dtype=torch.float32)
        }}

class BaseTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(10, 256)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(256, 4, batch_first=True), 4)
        self.output_projection = nn.Linear(256, 1)
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.output_projection(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

if __name__ == "__main__":
    model = BaseTransformer()
    train_loader = DataLoader(BasicDataset('train'), batch_size=32)
    val_loader = DataLoader(BasicDataset('val'), batch_size=32)
    
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    trainer.fit(model, train_loader, val_loader)
    
    # Save results
    import json
    results = {{"task": "{task['name']}", "status": "completed_fallback"}}
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
'''
    
    def _generate_model_code(self, task: Dict, dimensions: Dict) -> str:
        """Generate model code with correct dimensions for the task"""
        return f'''import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
import math
from output_shape import output_shape

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        input_dim={dimensions['input_dim']},
        output_dim={dimensions['output_dim']},
        d_model={dimensions['suggested_d_model']},
        nhead=4,
        num_layers={dimensions['suggested_layers']},
        dim_feedforward={dimensions['suggested_d_model'] * 4},
        dropout=0.1,
        learning_rate=1e-4,
        max_steps={dimensions['estimated_training_steps']},
        task_type='{task['task_type']}',
        debug=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.debug = debug
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        if task_type == 'classification':
            self.loss_fn = nn.CrossEntropyLoss()
        elif task_type == 'regression':
            self.loss_fn = nn.MSELoss()
        else:  # sequence
            self.loss_fn = nn.CrossEntropyLoss()
        
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.max_steps = max_steps

    @output_shape
    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        x = self.transformer(x, src_key_padding_mask=mask)
        
        if self.task_type == 'sequence':
            output = self.output_projection(x)
        else:
            x = x.mean(dim=1)  # Global average pooling
            output = self.output_projection(x)
        
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']
        mask = batch.get('mask', None)
        
        y_hat = self(x, mask)
        
        if self.task_type == 'classification' and y.dtype == torch.long:
            loss = self.loss_fn(y_hat, y)
            acc = (y_hat.argmax(dim=-1) == y).float().mean()
            self.log('train_acc', acc, prog_bar=True)
        else:
            loss = self.loss_fn(y_hat, y)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['input'], batch['target']
        mask = batch.get('mask', None)
        
        y_hat = self(x, mask)
        
        if self.task_type == 'classification' and y.dtype == torch.long:
            loss = self.loss_fn(y_hat, y)
            acc = (y_hat.argmax(dim=-1) == y).float().mean()
            self.log('val_acc', acc, prog_bar=True)
        else:
            loss = self.loss_fn(y_hat, y)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        print(f"Epoch {{self.current_epoch}} completed")

    def on_validation_epoch_end(self):
        print(f"Validation epoch {{self.current_epoch}} completed")
'''

    def _generate_dataloader_code(self, task: Dict) -> str:
        """Generate dataloader code for the task"""
        return f'''import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset

class TaskDataset(Dataset):
    def __init__(self, split='train'):
        # Generate sequence data for transformer
        # Create synthetic time series with multiple features
        num_samples = 1000 if split == 'train' else 200
        seq_len = 32  # sequence length for transformer
        num_features = 10  # input features per timestep
        
        # Generate sequential data (num_samples, seq_len, num_features)
        self.data = np.random.randn(num_samples, seq_len, num_features)
        
        # Generate target values (regression task)
        self.labels = np.random.randn(num_samples, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {{
            'input': torch.tensor(self.data[idx], dtype=torch.float32),  # Shape: (seq_len, num_features)
            'target': torch.tensor(self.labels[idx], dtype=torch.float32)  # Shape: (1,)
        }}

def get_dataloaders(batch_size=32, num_workers=0):
    train_dataset = TaskDataset('train')
    val_dataset = TaskDataset('val')
    test_dataset = TaskDataset('test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
'''
    
    def _get_task_dimensions(self, task: Dict) -> Dict:
        """Get model dimensions for the task"""
        return {
            'input_dim': 10,  # features per timestep in sequence
            'output_dim': 1,  # single regression output
            'suggested_d_model': 256,  # transformer hidden dimension
            'suggested_layers': 4,  # transformer layers
            'estimated_training_steps': 1000
        }
    
    def _generate_train_script(self, task: Dict, dimensions: Dict) -> str:
        return f"""import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
import torch
from torch.utils.data import DataLoader
import sys
import os
import wandb
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
    
    # Initialize WandB
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        
        # WandB Logger
        wandb_logger = WandbLogger(
            project="predictors-framework",
            name=f"{task['name']}",
            tags=["automated-discovery", "{task['task_type']}"],
            log_model=False,  # Don't upload model artifacts to save costs
            save_dir="logs"
        )
        
        # Log task metadata
        wandb_logger.experiment.config.update({{
            "task_name": "{task['name']}",
            "task_type": "{task['task_type']}",
            "evaluation_metric": "{task.get('evaluation_metric', 'unknown')}",
            "expected_baseline": "{task.get('expected_baseline', 'unknown')}",
            "input_dim": {dimensions['input_dim']},
            "output_dim": {dimensions['output_dim']},
            "d_model": {dimensions['suggested_d_model']},
            "num_layers": {dimensions['suggested_layers']},
            "hypothesis": "{task.get('hypothesis', 'unknown')}"
        }})
        
        logger = wandb_logger
    else:
        print("⚠️ WandB API key not found, using CSV logger")
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
    
    # Finish WandB run
    if wandb_api_key and wandb.run is not None:
        wandb.finish()
    
    print("\\nTraining complete!")
    print(f"Results saved to results.json")
    if wandb_api_key:
        print("🌊 Training logged to WandB")

if __name__ == '__main__':
    main()
"""