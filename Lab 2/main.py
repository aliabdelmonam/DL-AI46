from model import Model
from data_preprocessing import data_split

import torch
import torch.nn as nn
from torch.optim import Optimizer

(X_train_tensor, X_validation_tensor, y_train_tensor, y_validation_tensor),(test_data) = data_split()

model = Model(X_train_tensor.shape)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


print("Starting model training...")
model.fit(
    x=X_train_tensor,
    y=y_train_tensor,
    criterion=criterion,
    optimizer=optimizer,
    epochs=100,
    x_val=X_validation_tensor,
    y_val=y_validation_tensor
)
print("Training complete.")

