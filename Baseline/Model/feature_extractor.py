import torch
import torch.nn as nn


def calculate_padding(kernel_size):
    # padding = same means that input tensor size and output tensor size would be the same
    # out_tensor_size=nout, in_tensor_size=nin, kernel_size(dimension)=f, stride=s
    # nout = (nin + 2p - f)//s + 1
    # p = (f-1)//2, given that, nout = nin.
    return (kernel_size-1)//2

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=128, kernel_size=8, padding=calculate_padding(8), bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=calculate_padding(5), bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=calculate_padding(3), bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        # the shape of x was [batch_size, 3, 1, 128],
        # and initially it was showing error, 1 is a dummy axis, thus removed
        x = x.squeeze(2)
        return self.model(x)
        
        

class ContrastiveHead(nn.Module):
    def __init__(self, contrastive_out=128):
        super(ContrastiveHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(in_features=128, out_features=contrastive_out)
        )
    
    def forward(self, x):
        x = x.reshape(-1, 128*1)
        c = self.head(x)
        return c
