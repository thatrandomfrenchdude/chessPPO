import torch.nn as nn

class ChessPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Simpler architecture with better numerical stability
        self.net = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.LayerNorm(4096)
        )

    def forward(self, x):
        return self.net(x)

class ChessValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Input processing
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Hidden layers
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Value head
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)