import torch
import torch.nn.functional as F

def binary_log_likelihood(y, log_y_hat):
    # standard LL for vectors of binary labels y and log predictions log_y_hat
    return (y * -F.softplus(-log_y_hat)) + (1 - y) * (-log_y_hat - F.softplus(-log_y_hat))

def H(x):
    # Heaviside function, 0 if x < 0 else 1
    return torch.div(F.threshold(x, 0, 0), x)

def center(x):
  mu = (torch.sum(x) - x) / torch.FloatTensor(x.shape[0] - 1)
  return x - mu

def safe_log_prob(x, eps=1e-8):
  return torch.log(torch.clamp(x, eps, 1.0))