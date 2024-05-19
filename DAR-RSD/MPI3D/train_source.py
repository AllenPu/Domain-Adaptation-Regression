import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import model
import transform as tran
import os
import argparse


#
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='PyTorch DAregre experiment')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--src', type=str, default='rc', metavar='S',
                    help='source dataset')
parser.add_argument('--tgt', type=str, default='t', metavar='S',
                    help='target dataset')
parser.add_argument('--lr', type=float, default=0.1,
                        help='init learning rate for fine-tune')
parser.add_argument('--gamma', type=float, default=0.0001,
                        help='learning rate decay')
parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
parser.add_argument('--tradeoff', type=float, default=0.001,
                        help='tradeoff of RSD')
parser.add_argument('--tradeoff2', type=float, default=0.01,
                        help='tradeoff of BMP')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
# set dataset
batch_size = {"train": 36, "val": 36, "test": 4}
rc="realistic.txt"
rl="real.txt"
t="toy.txt"

rc_t="realistic_test.txt"
rl_t="real_test.txt"
t_t="toy_test.txt"

if args.src =='rl':
    source_path = rl
elif args.src =='rc':
    source_path = rc
elif args.src =='t':
    source_path = t


if args.tgt =='rl':
    target_path = rl
elif args.tgt =='rc':
    target_path = rc
elif args.tgt =='t':
    target_path = t

if args.tgt =='rl':
    target_path_t = rl_t
elif args.tgt =='rc':
    target_path_t = rc_t
elif args.tgt =='t':
    target_path_t = t_t