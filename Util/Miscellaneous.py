import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    z = mu + eps*std
    return z, mu, std

# weight init
def weights_init(m):
    init_type="xavier_uniform"
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform(m.weight.data, 1.)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
# # Have to modify it
# def f_recon(img, netE, netD, zdim, mode="train", clipping=False):
#     bs, imsize = img.shape[0], img.shape[2]
#     mu_logvar = netE(img).view(bs,-1)
#     mu = mu_logvar[:,0:zdim]
#     logvar = mu_logvar[:,zdim:]
#     z = reparameterize(mu, logvar)
    
#     if mode=="train":
#         if clipping: z = torch.clamp(z, min=-1.0, max=1.0)
#         recon = netD(z).view(bs,3,imsize,imsize)
#     else:
#         if clipping: mu = torch.clamp(mu, min=-1.0, max=1.0)
#         recon = netD(mu).view(bs,3,imsize,imsize)
#     return recon, mu, logvar

def nt_xent_loss(out_1, out_2, temperature=0.5):
    # Flatten out_1 and out_2
    out_1 = out_1.view(out_1.size(0), -1)
    out_2 = out_2.view(out_1.size(0), -1)
    
    # Normalize the vectors
    epsilon = 1e-8
    out_1 = F.normalize(out_1, dim=-1, p=2)
    out_2 = F.normalize(out_2, dim=-1, p=2)
    
    # Concatenate the normalized outputs
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)
    
    # Full similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)
    
    # Mask to remove self-similarities
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    
    # Sum over all negative similarities
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)
    
    # Positive similarity
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    
    # Create the positive similarities for both pairs
    pos = torch.cat([pos, pos], dim=0)
    
    # Compute the loss
    loss = -torch.log(pos / (neg + epsilon)).mean()
    
    return loss








class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
