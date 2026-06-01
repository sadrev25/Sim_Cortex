import torch
import torch.nn as nn

class RiskHead(nn.Module):
    def __init__(self, latent_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, latent):
        return self.net(latent).squeeze(-1)
