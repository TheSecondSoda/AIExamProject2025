import torch
import torch.nn as nn

def get_metadata_model(input_dim=12):
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)  # Single output: the Pawpularity score
    )
    return model