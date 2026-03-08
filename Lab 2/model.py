import torch
import torch.nn as nn
from typing import OrderedDict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim import Optimizer
import torch.optim.lr_scheduler
import os

class Model(nn.Module):
  def __init__(self,shape) -> None:
    super().__init__()
    self.shape = shape
    self.layers = nn.Sequential(OrderedDict([
        ("linear1",nn.Linear(self.shape[1],64)),
        ("relu1",nn.ReLU()),
        ("dropout1",nn.Dropout(p=0.5)),
        ("linear2",nn.Linear(64,32)),
        ("output",nn.Linear(32,1))
    ]))

  def forward(self, x):
    return self.layers(x)

  def fit(self, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module, optimizer: Optimizer, epochs: int = 100,
           reset_weights: bool = False, x_val: torch.Tensor = None, y_val: torch.Tensor = None,
              metric_fn = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None, 
              clip_value: float = None,patience: int = None, min_delta: float = 0.0,
              save_path: str = None, save_best_only: bool= True, save_every_n_epochs: int =0):
    if reset_weights:
      print("Resetting model weights for a fresh start.")
      for m in self.modules():
          if isinstance(m, nn.Linear):
              nn.init.xavier_uniform_(m.weight) # Kaiming initialization
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0) # Initialize biases to zero

    # Early Stopping initialization
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        self.train() # Set model to training mode
        optimizer.zero_grad()  # Zero the gradients
        train_logits = self.forward(x)
        train_loss = criterion(train_logits, y.unsqueeze(1)) # Ensure y has same dimensions as logits
        train_loss.backward()  # Perform backward pass
        if clip_value is not None:
          torch.nn.utils.clip_grad_norm_(self.parameters(), clip_value) # Apply gradient clipping
        optimizer.step()  # Update weights

        if scheduler:
            scheduler.step() # Step the learning rate scheduler
        log_message = f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}'

        if metric_fn:
            train_metric = metric_fn(train_logits, y.unsqueeze(1))
            log_message += f', Train Metric: {train_metric.item():.4f}'

        if x_val is not None and y_val is not None:
          self.eval() # Set model to evaluation mode
          with torch.no_grad(): # Disable gradient calculations for validation
            val_logits = self.forward(x_val)
            val_loss = criterion(val_logits, y_val.unsqueeze(1))
            log_message += f', Val Loss: {val_loss.item():.4f}'
            if metric_fn:
                val_metric = metric_fn(val_logits, y_val.unsqueeze(1))
                log_message += f', Val Metric: {val_metric.item():.4f}'
          self.train() # Set model back to training mode

          # Early stopping logic
          if val_loss.item() < best_val_loss - min_delta:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_model_state = self.state_dict() # Save best model state

            if save_path and save_best_only:
              os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True) # Create directory if it doesn't exist
              torch.save(best_model_state, save_path)
              print(f"Best model checkpoint saved to {save_path}")
          else:
            patience_counter += 1
            if patience is not None and patience_counter >= patience:
              print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss.")
              if best_model_state:
                self.load_state_dict(best_model_state) # Load best model weights
                print("Loaded best model weights.")
              break # Exit training loop

        # Periodic checkpointing (independent of best_val_loss and save_best_only)
        if save_path and not save_best_only and save_every_n_epochs > 0 and (epoch + 1) % save_every_n_epochs == 0:
          base_checkpoint_dir = os.path.dirname(save_path)
          periodic_sub_dir = os.path.join(base_checkpoint_dir, 'periodic_saves')
          os.makedirs(periodic_sub_dir, exist_ok=True) # Create periodic saves directory

          base_filename_stem, ext = os.path.splitext(os.path.basename(save_path))
          periodic_filename = os.path.join(periodic_sub_dir, f"{base_filename_stem}_epoch_{epoch+1}{ext if ext else '.pth'}")

          torch.save(self.state_dict(), periodic_filename)
          print(f"Periodic model checkpoint saved to {periodic_filename}")

        if (epoch + 1) % 5 == 0:
            print(log_message)

  def evaluate(self, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module, metric_fn = None):
    self.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
      logits = self.forward(x)
      loss = criterion(logits, y.unsqueeze(1))
      if metric_fn:
        metric = metric_fn(logits, y.unsqueeze(1))
        return loss.item(), metric.item()
      return loss.item()