# Toy problem from REBAR paper
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pdb

# minibatch size
M = 32
steps = 8000
rs = 1337
eta = 0.1
dx = 0.001
torch.manual_seed(rs)
check_gradients = False

targets = torch.FloatTensor([0.45]).repeat(M)
targets = Variable(targets, requires_grad=False)

def compute_gradient_check(f_plus, f_minus):
  return (f_plus - f_minus) / (2 * dx)

def binary_log_likelihood(y, log_y_hat):
  return (y * -F.softplus(-log_y_hat)) + (1 - y) * (-log_y_hat - F.softplus(-log_y_hat))

def H(z):
  # Heaviside function
  return torch.div(F.threshold(z, 0, 0), z)

def relax(z, l=0.5):
  return F.sigmoid(z / l)

def reparam_g(v, b, theta):
  # z_tilde =
  # log(v/1-v * 1/1-theta + 1) if b = 1
  # -log(v/1-v 1/theta + 1) if b = 0
  # relax(z_tilde)
  g = (b * F.softplus(torch.log(v) - torch.log((1 - v) * (1 - theta)))) \
      + ((1 - b) * (-F.softplus(torch.log(v) - torch.log(v * (1 - theta)))))
  return relax(g)

def estimators(u, v, theta):
  z = (torch.log(theta) - torch.log(1 - theta)) + (torch.log(u) - torch.log(1 - u))
  hz = H(z)
  sz = relax(z) + 1e-9
  gz = reparam_g(v, hz, theta) + 1e-9
  return z, hz, sz, gz

# create model
class SimpleBernoulli(nn.Module):
  def __init__(self, init_value=0.5):
    super(SimpleBernoulli, self).__init__()
    self.w = nn.Parameter(torch.FloatTensor([init_value]))
    self.nonlin = nn.Sigmoid()
  
  def forward(self):
    # Returns number between 0 and 1
    return self.nonlin(self.w)

soft_concrete_model = SimpleBernoulli()
rebar_model = SimpleBernoulli()
soft_concrete_opt = optim.Adam(soft_concrete_model.parameters(), lr=1e-3)
rebar_opt = optim.Adam(rebar_model.parameters(), lr=1e-3)
mse = nn.MSELoss()

# sample a minibatch for a single-sample montecarlo estimate
for i in range(steps+1):
  sc_theta = soft_concrete_model.forward().repeat(M)
  rebar_theta = rebar_model.forward().repeat(M)

  # sample (u,v) ~ Unif(0,1) M times
  uv = Variable(torch.FloatTensor(2, M).uniform_(0, 1), requires_grad=False)
  u = uv[0] + 1e-9 # for numerical stability
  v = uv[1] + 1e-9 # for numerical stability

  _, sc_hz, sc_sz, _ = estimators(u, v, sc_theta)
  hard_concrete_loss = mse(sc_hz, targets)
  soft_concrete_loss = mse(sc_sz, targets)

  soft_concrete_opt.zero_grad()
  soft_concrete_loss.backward()
  soft_concrete_opt.step()

  # rebar estimate
  z, r_hz, r_sz, r_gz = estimators(u, v, rebar_theta)
  # grad p(b) :- d/d_theta log p(b = H(z))
  grad_nllP = torch.autograd.grad(binary_log_likelihood(r_hz, torch.log(rebar_theta)).split(1), rebar_theta, retain_graph=True, create_graph=True)[0]
  f_hz = (r_hz - targets) ** 2
  f_sz = (r_sz - targets) ** 2
  f_gz = (r_gz - targets) ** 2
  grad_f_sz = torch.autograd.grad(f_sz.split(1), rebar_theta, retain_graph=True, create_graph=True)[0]
  grad_f_gz = torch.autograd.grad(f_gz.split(1), rebar_theta, retain_graph=True, create_graph=True)[0]

  rebar_estimator = ((f_hz.detach() - eta * f_gz) * -grad_nllP + eta * grad_f_sz - eta * grad_f_gz).mean()

  if (check_gradients):
    # check gradients
    rebar_theta_pve = rebar_theta.detach() + dx
    rebar_theta_mve = rebar_theta.detach() - dx
    _, __, r_sz_pve, r_gz_pve = estimators(u, v, rebar_theta_pve)
    _, __, r_sz_mve, r_gz_mve = estimators(u, v, rebar_theta_mve)
    #f_hz_pve = (r_hz_pve - targets) ** 2
    f_sz_pve = (r_sz_pve - targets) ** 2
    f_gz_pve = (r_gz_pve - targets) ** 2
    #f_hz_mve = (r_hz_mve - targets) ** 2
    f_sz_mve = (r_sz_mve - targets) ** 2
    f_gz_mve = (r_gz_mve - targets) ** 2
    grad_f_sz_check = compute_gradient_check(f_sz_pve, f_sz_mve)
    grad_f_gz_check = compute_gradient_check(f_gz_pve, f_gz_mve)

    pdb.set_trace()

    grad_f_sz_check_norm = torch.norm(grad_f_sz_check - grad_f_sz) / (torch.norm(grad_f_sz) + torch.norm(grad_f_sz_check))
    grad_f_gz_check_norm = torch.norm(grad_f_gz_check - grad_f_gz) / (torch.norm(grad_f_gz) + torch.norm(grad_f_gz_check))

  rebar_opt.zero_grad()
  rebar_estimator.backward()
  rebar_opt.step()

  # print "r_sz", r_sz
  # print "r_gz", r_gz
  # print "rebar_estimator", rebar_estimator
  # non-differentiable, stochastic function we actually want to optimize
  discrete = torch.bernoulli(rebar_theta.detach())
  rebar_loss = mse(discrete, targets)

  if i % 100 == 0:
    print "step: ", i
    print "hard_concrete_loss", hard_concrete_loss
    print "soft_concrete_loss", soft_concrete_loss
    print "rebar_estimator", rebar_estimator
    #print "rebar_z", z
    print "rebar_loss", rebar_loss
    print "rebar_theta", rebar_theta[0].data