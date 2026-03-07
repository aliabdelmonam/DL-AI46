import torch
import torch.nn as nn
from typing import OrderedDict
from torch.optim import Optimizer
import torch.optim.lr_scheduler

class Model(nn.Module):
  def __init__(self,shape) -> None:
    super().__init__()
    self.shape = shape
    self.layers = nn.Sequential(OrderedDict([
        ("linear1",nn.Linear(self.shape[1],64)),
        ("relu1",nn.ReLU()),
        ("linear2",nn.Linear(64,32)),
        ("relu2",nn.ReLU()),
        ("output",nn.Linear(16,1))
    ]))

  def forward(self, x):
    return self.layers(x)

  def fit(self, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module, optimizer: Optimizer, epochs: int = 100,\
           reset_weights: bool = False, x_val: torch.Tensor = None, y_val: torch.Tensor = None,\
              metric_fn = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None, \
              clip_value: float = None):
    if reset_weights:
      print("Resetting model weights for a fresh start.")
      for m in self.modules():
          if isinstance(m, nn.Linear):
              nn.init.xavier_uniform_(m.weight) # Kaiming initialization
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0) # Initialize biases to zero

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

        if (epoch + 1) % 10 == 0:
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