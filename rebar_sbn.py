
# coding: utf-8

# Implementation of REBAR (https://arxiv.org/abs/1703.07370), a low-variance, unbiased gradient estimator for discrete latent variable models. This notebook is focused on the generative modeling experiments on the MNIST and Omniglot datasets from Section 5.2.1.
# 
# The problem being solved is $\text{max} \hspace{5px} \mathbb{E} [f(b, \theta) | p(b) ]$, $b$ ~ Bernoulli($\theta$).
# 
# For generative modeling, the objective is to maximize a single-sample variational lower bound on the log-likelihood. There are two networks, one to model $q(b|x,\theta)$ and one to model $p(x,b|\theta)$. The former is the variational distribution and the latter is the joint probability distribution over the data and latent stochastic variables $b$.
# 
# The **ELBO**, or evidence lower bound which we seek to maximize, is: 
# 
# $$
# \log p(x \vert \theta) \geq \mathbb{E}_{q(b \vert x,\theta)} [ \log p(x,b\vert\theta) - \log q(b \vert x,\theta)]
# $$
# 
# In practice, the Q-network has its own set of parameters $\phi$ and the generator network $P$ has its own parameters $\theta$.
# 
# I'll refer to the learning signal $\log p(x,b\vert\theta) - \log q(b \vert x,\theta)$ as $l(x,b)$ for shorthand.
# 
# The following is an implementation of a Sigmoid Belief Network (SBN) with REBAR gradient updates. I tried to follow the [author's TensorFlow implementation](https://github.com/tensorflow/models/blob/master/research/rebar/rebar.py) closely; there are a lot of computational statistics stuff going on that need to be implemented carefully.
# 
# For an in-depth treatment on SBNs, see [this paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.1777&rep=rep1&type=pdf) by R. Neal.

# We're just going to focus on the nonlinear SBN REBAR model.
# The model is pretty complex, so I'll implement it as separate modules and try to explain them
# one by one.

# In[1]:

import pdb
import functools
import h5py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.autograd import grad
import matplotlib.pyplot as plt
import numpy as np

import rebar.datasets as datasets
import rebar.util as U

from tqdm import tqdm

# In[2]:

# Some global parameters we'll need later
hparams = {
    'model': 'SBNRebar',
    'learning_rate':3e-4,
    'n_hidden':200,
    'n_input':784,
    'temperature':0.5,
    'eta':1.0,
    'batch_size':24,
    'task':'sbn',
    'n_layers': 1,
    'dynamic_b': False,
    'ema_beta': 0.999,
    'train_steps': 1000000,
    'log_every': 1000,
    'save_every': 100000,
    'random_seed': 12321
}


# We'll define samplers for producing the "hard" and "soft" reparameterized samples needed for computing the REBAR gradient.

# In[3]:

def random_sample(log_alpha, u, layer, uniform_samples_v, temperature=None):
    """Returns sampled random variables parameterized by log_alpha."""
    # Generate tied randomness for later
    if layer not in uniform_samples_v:
        uniform_samples_v[layer] = u_to_v(log_alpha, u)
        
    # Sample random variable underlying softmax/argmax
    x = log_alpha + U.safe_log_prob(u) - U.safe_log_prob(1 - u)
    samples = ((x > 0).float()).detach()

    return {
        'preactivation': x,
        'activation': samples,
        'log_param': log_alpha,
    }, uniform_samples_v

def random_sample_soft(log_alpha, u, layer, uniform_samples_v, temperature=None):
    """Returns sampled random variables parameterized by log_alpha."""

    # Sample random variable underlying softmax/argmax
    x = (log_alpha + U.safe_log_prob(u) - U.safe_log_prob(1 - u)) / temperature.view(-1)
    y = F.sigmoid(x)

    return {
        'preactivation': x,
        'activation': y,
        'log_param': log_alpha
    }, uniform_samples_v

def random_sample_soft_v(log_alpha, _, layer, uniform_samples_v, temperature=None):
    """Returns sampled random variables parameterized by log_alpha."""
    v = uniform_samples_v[layer]
    return random_sample_soft(log_alpha, v, layer, uniform_samples_v, temperature)


# This next bit, for producing common random numbers, is for variance reduction. [The general idea behind common random numbers is easy enough to grasp](https://en.wikipedia.org/wiki/Variance_reduction), but what the authors are doing here is a bit more subtle. According to Appendix G.2, they're correlating u and v to reduce the variance of the gradient by first sampling u and then using that to determine v.

# In[4]:

# Random samplers
def u_to_v(log_alpha, u, eps = 1e-8):
   """Convert u to tied randomness in v."""
   u_prime = F.sigmoid(-log_alpha)  # g(u') = 0
   v_1 = (u - u_prime) / torch.clamp(1 - u_prime, eps, 1)
   v_1 = torch.clamp(v_1.clone(), 0, 1).detach()
   v_1 = v_1.clone()*(1 - u_prime) + u_prime
   v_0 = u / torch.clamp(u_prime, eps, 1)
   v_0 = torch.clamp(v_0.clone(), 0, 1).detach()
   v_0 = v_0.clone() * u_prime
   v = u.clone()
   v[(u > u_prime).detach()] = v_1[(u > u_prime).detach()]
   v[(u <= u_prime).detach()] = v_0[(u <= u_prime).detach()]
   # TODO: add pytorch check
   #v = tf.check_numerics(v, 'v sampling is not numerically stable.')
   vv = v + (-v + u).detach()  # v and u are the same up to numerical errors
   return Variable(vv.data, requires_grad=False)


# This is the deterministic mapping we'll use to construct the stochastic layers of the Q- and P-networks.

# In[5]:

class Transformation(nn.Module):
    """
    Deterministic transformation between stochastic layers
    
        x -> FC -> Tanh -> FC -> Tanh() -> FC -> logQ
            
    """
    def __init__(self, n_input, n_hidden, n_output):
        super(Transformation, self).__init__()
        self.h = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_output))
        
        for layer in self.h:
            if hasattr(layer, 'weight'):
                U.scaled_variance_init(layer)
                
    def forward(self, x):
        return self.h(x)


# The RecognitionNet is the variational distribution (Q-network) and the GeneratorNet is the joint distribution of the data and latent variables (P-network). It looks like this for an unrolled 2-layer SBN, where Sample is the stochastic layer of Bernoulli units:
# 
# // Replace with figure?
# 
# x -> Transformation(x) -> Sample(x) -> Transformation(x) -> Sample(x)

# In[6]:

class RecognitionNet(nn.Module):
    """
    given x values, samples from Q and returns log Q(h|x)
    """
    def __init__(self, mean_xs, sampler):
        super(RecognitionNet, self).__init__()
        self.mean_xs = mean_xs
        self.sampler = sampler
        self.transforms = nn.ModuleList([Transformation(hparams['n_input'],
                                        hparams['n_hidden'], hparams['n_hidden'])])
        if hparams['n_layers'] > 1:
            for _ in range(1, hparams['n_layers']):
                self.transforms.append(Transformation(hparams['n_hidden'], hparams['n_hidden'],
                                                     hparams['n_hidden']))
        self.uniform_samples = dict()
        self.uniform_samples_v = dict()
        # generate randomness
        for i in range(hparams['n_layers']):
            self.uniform_samples[i] = Variable(
                torch.FloatTensor(hparams['batch_size'], hparams['n_hidden']).uniform_(0,1),
                requires_grad=False)
            
    def forward(self, x, sampler_=None):
        if sampler_ is not None:
            sampler = sampler_
        else:
            sampler = self.sampler
        samples = {}
        samples[-1] = {'activation': x}
        # center the input
        samples[-1]['activation'] = samples[-1]['activation'].clone() - self.mean_xs
        samples[-1]['activation'] = (samples[-1]['activation'].clone() + 1)/2.
        logQ = []
        logitss = []
        for i,t in enumerate(self.transforms):
            input = 2 * samples[i-1]['activation'] - 1.0
            logits = t(input)
            logitss.append(logits)
            # expect sampler to return a dictionary with key 'activation'
            samples[i], self.uniform_samples_v = sampler(logits, self.uniform_samples[i],
                                                         i, self.uniform_samples_v)
            logQ.append(U.binary_log_likelihood(samples[i]['activation'], logits))  
        # logQHard, samples
        return logQ, samples, logitss

class GeneratorNet(nn.Module):
    """
    Returns learning signal and function. Reconstructs the input.
    """
    def __init__(self, mean_xs):
        super(GeneratorNet, self).__init__()
        self.transforms = []
        for i in range(hparams['n_layers']):
            if i == 0:
                n_output = hparams['n_input']
            else:
                n_output = hparams['n_hidden']
            self.transforms.append(Transformation(hparams['n_hidden'],
                                                 hparams['n_hidden'], n_output))
        self.transforms = nn.ModuleList(self.transforms)
        self.prior = nn.Parameter(torch.zeros(hparams['n_hidden']))
        self.train_bias = -np.log(1./np.clip(mean_xs.data.numpy(), 0.001, 0.999)-1.).astype(np.float32)
        self.train_bias = Variable(torch.from_numpy(self.train_bias).float(), requires_grad=False)
        
    def forward(self, x, samples, logQ):
        """
        Args:
            samples: dictionary of sampled latent variables
            logQ: list of log q(h_i) terms
        """
        sum_logQ = torch.sum(torch.stack(logQ), 0)
        logPPrior = U.binary_log_likelihood(samples[hparams['n_layers']-1]['activation'], self.prior)
        for i in reversed(range(hparams['n_layers'])):
            # Set up the input to the layer
            input = 2 * samples[i]['activation'] - 1.0
            h = self.transforms[i](input)
            if i == 0:
                logP = U.binary_log_likelihood(x, h + self.train_bias)
            else:
                logPPrior = logPPrior + U.binary_log_likelihood(samples[i-1]['activation'], h)
        # Note that logP(x,b) = logP(b|x) + logP(x)
        # reinforce_learning_signal (l(x,b)), reinforce_model_grad
        debug = {
            'logP': logP,
            'logPPrior': logPPrior,
            'samples': samples
        }
        return logP + logPPrior - sum_logQ, logP + logPPrior, debug         


# Now we can put these modules together inside the SBNRebar module

# In[7]:

class SBNRebar(nn.Module):
    def __init__(self, mean_xs):
        super(SBNRebar, self).__init__()
        self.mean_xs = mean_xs   
        self._temperature = Variable(torch.FloatTensor([hparams['temperature']]), requires_grad=False)
        self.recognition_network = RecognitionNet(mean_xs, random_sample)
        self.generator_network = GeneratorNet(mean_xs) 
        self.eta = Variable(torch.FloatTensor([hparams['eta']]), requires_grad=False)
                
    def multiply_by_eta(self, grads):
        res = []
        for g in grads:
            res.append(g*self.eta)
        return res

    def forward(self, x):
        """
        All of the passes through the Q- and P-networks are here
        """
        ###################################
        # REINFORCE step (compute ELBO, etc.)
        ###################################
        # hardELBO is the non-differentiable learning signal, l(x,b)
        #
        # reinforce_model_grad is the joint distribution of interest p(x,b,\theta), 
        #   and the gradient of l(x,b) wrt the P-network parameters is grad E[logP + logPPrior]  
        #   = grad E[reinforce_model_grad]
        # 
        # See https://github.com/tensorflow/models/blob/master/research/rebar/rebar.py#L716
        logQHard, hardSamples, logits = self.recognition_network(x)
        hardELBO, reinforce_model_grad, debug = self.generator_network(x, hardSamples, logQHard)
        
        ###################################
        # compute Gumbel control variate
        ###################################
        # See https://github.com/tensorflow/models/blob/master/research/rebar/rebar.py#L659
        logQ, softSamples, _ = self.recognition_network(x, sampler_=functools.partial(
            random_sample_soft, temperature=self._temperature))
        softELBO, _, _ = self.generator_network(x, softSamples, logQ)
        # compute softELBO_v (same value as softELBO, different grads) :- zsquiggle = g(v, b, \theta)
        # NOTE: !!! Because of the common random numbers (u_to_v), z is distributed as z|b. 
        # So the reparameterization for p(z|b) is just g(v,b,\theta) == g(v,\theta) == log(\theta/1-\theta) + log(v/1-v)
        # This is why random_sample_soft_v() just calls random_sample_soft(). I'm 95% sure this is correct...        
        logQ_v, softSamples_v, _ = self.recognition_network(x, sampler_=functools.partial(
            random_sample_soft_v, temperature=self._temperature))
        # should be the same value as softELBO but different grads
        softELBO_v, _, _ = self.generator_network(x, softSamples_v, logQ_v)
        gumbel_cv_learning_signal = softELBO_v.detach()
        gumbel_cv = gumbel_cv_learning_signal * torch.sum(torch.stack(logQHard), 0) - softELBO + softELBO_v

        return {
            'logQHard': logQHard,
            'hardELBO': hardELBO,
            'reinforce_model_grad': reinforce_model_grad,
            'gumbel_cv': gumbel_cv,
            'logits': logits,
            'hardSamples': hardSamples,
            'debug': debug
        }   

class Baseline(nn.Module):
    def __init__(self, mean_xs):
        super(Baseline, self).__init__()
        # For centering the learning signal, from the NVIL paper (2.3.1) https://arxiv.org/pdf/1402.0030.pdf
        # Input dependent baseline that is trained to minimize the MSE with the learning signal
        self.means = mean_xs
        self.out = nn.Sequential(
           nn.Linear(hparams['n_input'], 100),
           nn.Tanh(),
           nn.Linear(100, 1))
    
        for layer in self.out:
            if hasattr(layer, 'weight'):
                U.scaled_variance_init(layer)
                
    def forward(self, x):
        x = x - self.means
        return self.out(x).squeeze()


if __name__ == '__main__':
    # In[8]:

    # Random seed
    torch.manual_seed(hparams['random_seed'])

    # Load MNIST dataset
    train_xs, val_xs, test_xs = datasets.load_data(hparams)
    # create Dataloader
    train_dataloader = DataLoader(train_xs, shuffle=True, batch_size=hparams['batch_size'], drop_last=True, num_workers=0)
    # mean centering on training data
    mean_xs = Variable(torch.from_numpy(np.mean(train_xs, axis=0)).float(), requires_grad=False)

    # In[ ]:

    sbn = SBNRebar(mean_xs)

    baseline = Baseline(mean_xs)
    baseline_loss = nn.MSELoss()
    sbn_opt = optim.Adam(sbn.parameters(), lr=hparams['learning_rate'], betas=(0.9, 0.99999))
    baseline_opt = optim.Adam(baseline.parameters(), lr=10*hparams['learning_rate'])

    # The main training loop, where we compute REBAR gradients and update model parameters

    # In[ ]:

    # Exponential Moving Average for log variance calculation
    ema_first_moment = 0.
    ema_second_moment = 0.
    beta = hparams['ema_beta']
    log_every = hparams['log_every']
    save_every = hparams['save_every']
    n = hparams['train_steps']

    scores = []
    save_dir = "/tmp/rebar/{}".format(hparams['random_seed'])
    try:
        os.makedirs(save_dir)
    except:
        pass
    scores_file = h5py.File(os.path.join(save_dir, 'scores.hdf5'), 'w')

    step = 0.
    while step < n:
        lHats = []
        log_grad_variances = []
        for x in tqdm(train_dataloader):
            x = Variable(x, requires_grad=False)
            sbn_outs = sbn.forward(x)
            baseline_out = baseline.forward(x)

            nvil_gradient = (sbn_outs['hardELBO'].detach() - baseline_out) * \
                    torch.sum(torch.stack(sbn_outs['logQHard']), 0) + sbn_outs['reinforce_model_grad']

            f_grads = grad(-nvil_gradient.mean(), sbn.parameters(), retain_graph=True)
            gumbel_grads = grad(sbn_outs['gumbel_cv'].mean(), sbn.parameters())
            h_grads = sbn.multiply_by_eta(gumbel_grads)
            total_grads = [(g_a + g_b) for (g_a, g_b) in zip(f_grads, h_grads)]

            # training objective
            lhat = sbn_outs['hardELBO'].mean().detach()
            lHats.append(lhat.data[0])
            # baseline loss
            #baseline_y = baseline_loss(baseline_out, sbn_outs['hardELBO'].detach())

            # variance summaries
            first_moment = U.vectorize(total_grads, skip_none=True)
            second_moment = first_moment ** 2
            ema_first_moment = (beta * ema_first_moment) + (1 - beta) * first_moment
            ema_second_moment = (beta * ema_second_moment) + (1 - beta) * second_moment
            log_grad_variance = torch.log((ema_second_moment.mean() - (ema_first_moment.mean()) ** 2))
            log_grad_variances.append(log_grad_variance.data[0])

            sbn_opt.zero_grad()

            # set model grads with REBAR gradients
            for (g, p) in zip(total_grads, sbn.parameters()):
                p.grad = g
            sbn_opt.step()

            # update baseline
            # baseline_opt.zero_grad()        
            # baseline_y.backward()
            # baseline_opt.step()

            if step % log_every == 0: 
                #print('NVIL grads')
                #for f in f_grads:
                #    print(f.shape, f.mean().data[0], f.var().data[0])
                #print('Concrete grads')
                #for h in h_grads:
                #    print(h.shape, h.mean().data[0], h.var().data[0])
                print('step: {}, training objective (ELBO): {}, logGradVar: {}'.format(step, lhat.data[0], log_grad_variance.data[0]))
                print('grad ema first moment: {}'.format(ema_first_moment.mean().data[0]))
                #pdb.set_trace()
            if step % save_every == 0:
                torch.save(sbn, os.path.join(save_dir, 'sbn-step-{}.pt'.format(step)))
            step += 1
        scores.append((np.mean(lHats), np.mean(log_grad_variances)))
    scores_file.create_dataset('scores', data=scores)
    scores_file.close()
