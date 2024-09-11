import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

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
    
    
    
    
    
def activity_contrastive_loss(z, labels, temperature=0.5, eps=1e-8):
    # Normalize the latent vectors to ensure unit length
    z = F.normalize(z, dim=-1, p=2)
    
    # Full similarity matrix (N x N), where N is the batch size
    sim_matrix = torch.mm(z, z.t()) / temperature
    
    # Label matching matrix (1 if same activity, 0 otherwise)
    label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float().to(z.device)

    # Mask to ignore self-similarity
    mask = torch.eye(label_matrix.size(0), dtype=torch.bool).to(z.device)
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))  # Set diagonal to -inf to avoid self-similarity
    
    # Stabilize with log-sum-exp trick
    max_sim, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    exp_sim = torch.exp(sim_matrix - max_sim)  # Prevent overflow
    sum_exp_sim = exp_sim.sum(dim=1, keepdim=True) + eps  # Sum of all exponentials for normalization

    # Positive similarities (where labels match)
    pos_sim = exp_sim * label_matrix
    pos_sum_sim = pos_sim.sum(dim=1) + eps  # Sum of positive similarities
    
    # Negative similarities (where labels don't match)
    neg_sim = exp_sim * (1 - label_matrix)
    neg_sum_sim = neg_sim.sum(dim=1) + eps  # Sum of negative similarities
    
    # Positive and negative loss
    pos_loss = -torch.log(pos_sum_sim / sum_exp_sim)
    neg_loss = -torch.log(neg_sum_sim / sum_exp_sim)
    
    # Combine positive and negative loss
    loss = (pos_loss + neg_loss).mean()

    return loss





def pairwise_distance(x, y):

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, sigmas):

    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    beta = beta.to(x.get_device())
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix):

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost

def mmd_loss(source_features, target_features, device):

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    if device == 'gpu':
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.cuda.FloatTensor(sigmas))
        )
    else:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.FloatTensor(sigmas))
        )
    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
    loss_value = loss_value

    return loss_value

