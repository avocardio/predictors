import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import math


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
        input_dim,
        output_dim,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        learning_rate=1e-4,
        warmup_steps=1000,
        max_steps=100000,
        task_type='regression',  # 'regression', 'classification', 'sequence'
    ):
        super().__init__()
        self.save_hyperparameters()
        
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
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

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
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def on_train_epoch_end(self):
        print(f"Epoch {self.current_epoch} completed")

    def on_validation_epoch_end(self):
        print(f"Validation epoch {self.current_epoch} completed")