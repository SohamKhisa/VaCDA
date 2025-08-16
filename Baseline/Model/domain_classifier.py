
import torch
import torch.nn as nn
import torch.nn.functional as F

# num_domains = num_sources + 1(for target)
class DomainClassifier(nn.Module):
    def __init__(self, num_domains):
        super(DomainClassifier, self).__init__()
        self.num_domains = num_domains
        self.model = nn.Sequential(
            nn.Linear(in_features=128, out_features=500, bias=False),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(in_features=500, out_features=500, bias=False),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(in_features=500, out_features=self.num_domains)
        )

    def forward(self, x):
        x = x.reshape(-1, 128*1)
        logits = self.model(x)
        return logits
