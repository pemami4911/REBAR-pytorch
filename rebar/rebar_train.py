# author: Patrick Emami
import argparse
import os
from tqdm import tqdm 

import pprint as pp
import numpy as np

import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value

import datasets
import rebar

parser = argparse.ArgumentParser(description="")

# Data
parser.add_argument('--batch_size', default=128, help='')
parser.add_argument('--train_size', default=1000000, help='')
parser.add_argument('--val_size', default=10000, help='')

# Model cfg options here
parser.add_argument('--task', default='sbn', help='sbn|toy')
parser.add_argument('--model', default='SBNDynamicRebar')
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--n_layer', type=int, default=2)

# Training cfg options here
parser.add_argument('--n_epochs', type=int, default=1, help='')
parser.add_argument('--random_seed', type=int, default=24601, help='')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping')
parser.add_argument('--cuda', type=util.str2bool, default=True, help='')

# Misc
parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
parser.add_argument('--eval_freq', type=int,  default=20, help='How often to run eval')
parser.add_argument('--run_name', type=str, default='0')
parser.add_argument('--epoch_start', type=int, default=0, help='Restart at epoch #')
parser.add_argument('--save_model', type=util.str2bool, default=True, help='Save after epoch')
parser.add_argument('--load_path', type=str, default=' ')
parser.add_argument('--disable_tensorboard', type=util.str2bool, default=False)
parser.add_argument('--disable_progress_bar', type=util.str2bool, default=False)

args = vars(parser.parse_args())

# Pretty print the run args
pp.pprint(args)

# Set the random seed
torch.manual_seed(int(args['random_seed']))

# Optionally configure tensorboard
if not args['disable_tensorboard']:
    configure(os.path.join(args['log_dir'], args['task'], args['run_name']))


# Load the model parameters from a saved state
if args['load_path'] != '':
    print('  [*] Loading model from {}'.format(args['load_path']))

    model = torch.load(
        os.path.join(
            os.getcwd(),
            args['load_path']
        ))
# Instantiate model here
else:
    model = None

save_dir = os.path.join(os.getcwd(),
    'results',
    args['task'],
    args['run_name'])    

try:
    os.makedirs(save_dir)
except:
    pass

train_xs, valid_xs, test_xs = datasets.load_data(args)

training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
    shuffle=True, num_workers=4)

validation_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)

# Dataloaders, etc

#########################################
##          Training loop 
#########################################
epoch = int(args['epoch_start'])
step = epoch * args['train_size']
for i in range(epoch, epoch + int(args['n_epochs'])):

    # place in train mode
    model.train()

     for batch_id, sample in enumerate(tqdm(training_dataloader,
                disable=args['disable_progress_bar'])):
        sample = Variable(sample)
	if args['cuda']:
	    sample = sample.cuda()	

        # training loop stuff

        if not args['disable_tensorboard']:
            # log_value('avg_reward', R.mean().data[0], step)
        
        if step % int(args['log_step']) == 0:
            #print('epoch: {}, train_batch_id: {}, avg_reward: {}'.format(
            #    i, batch_id, R.mean().data[0]))
    
    print('  [*] starting validation')
    model.eval()

    for batch_id, val_sample in enumerate(tqdm(validation_dataloader,
            disable=args['disable_progress_bar'])):

        val_sample = Variable(val_sample)
	if args['cuda']:
	    val_sample = val_sample.cuda()

        if not args['disable_tensorboard']:
            #log_value('val_avg_reward', R[0].data[0], int(val_step))

        if val_step % int(args['log_step']) == 0:
    
    if args['save_model']:
        torch.save(model, os.path.join(save_dir, 'epoch-{}.pt'.format(i)))

    # optionally generate new training data
    # training_dataset = tsp_task.TSPDataset(train=True, size=size,
    #     num_samples=int(args['train_size']))
    # training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
    #     shuffle=True, num_workers=1)
