import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TaskClassifier, self).__init__()
        self.num_classes = num_classes
        
        self.model = nn.Sequential(
            nn.Linear(in_features=128, out_features=self.num_classes)
        )

    def forward(self, x):
        # because it has a three dimensional shape. which is not acceptable, because next layer is a dense layer.
        x = x.reshape(-1, 128*1)
        logits = self.model(x)
        return logits
