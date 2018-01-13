# Toy problem from REBAR paper
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# minibatch size
M = 512
steps = 8000
rs = 1337
torch.manual_seed(rs)

targets = torch.FloatTensor([0.45]).unsqueeze(1).repeat(1, M)
targets = Variable(targets, requires_grad=False)

def H(z):
  # Heaviside function
  return torch.div(F.threshold(z, 0, 0), z)

def relax(z, l=0.01):
  return F.sigmoid(z / l)

# create model
class SimpleBernoulli(nn.Module):
  def __init__(self, init_value=0.5):
    super(SimpleBernoulli, self).__init__()
    self.w = nn.Parameter(torch.FloatTensor([init_value]))
    self.nonlin = nn.Sigmoid()
  
  def forward(self):
    # Returns number between 0 and 1
    return self.nonlin(self.w)

# sample a minibatch for a single-sample montecarlo estimate
model = SimpleBernoulli()
opt = optim.Adam(model.parameters(), lr=1e-2)
mse = nn.MSELoss()

for i in range(steps+1):
  theta = model.forward()

  # sample (u,v) ~ Unif(0,1) M times
  uv = Variable(torch.FloatTensor(2, M).uniform_(0, 1))
  u = uv[0]
  v = uv[1]

  z = (torch.log(theta) - torch.log(1 - theta)) + (torch.log(u) - torch.log(1 - u))
  hz = H(z)
  sz = relax(z)
  # non-differentiable, stochastic function we actually want to optimize
  discrete = torch.bernoulli(theta.unsqueeze(1).repeat(1, M))

  hard_loss = mse(hz, targets)
  soft_loss = mse(sz, targets)
  discrete_loss = mse(discrete, targets)

  opt.zero_grad()
  soft_loss.backward()
  opt.step()

  if i % 1000 == 0:
    print "step: ", i
    print "hard_loss", hard_loss
    print "soft_loss", soft_loss
    print "discrete_loss", discrete_loss