import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import model
import transform as tran
import numpy as np
import os
import argparse
torch.set_num_threads(1)
import math
from read_data import ImageList
from tqdm import tqdm
import matplotlib.pyplot as  plt


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


#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_transforms = {
    'train': tran.rr_train(resize_size=224),
    'val': tran.rr_train(resize_size=224),
    'test': tran.rr_eval(resize_size=224),
}
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
    npz_path_source = '/home/rpu2/scratch/code/MPI3D_data/real.npz'
elif args.src =='rc':
    source_path = rc
    npz_path_source = '/home/rpu2/scratch/code/MPI3D_data/mpi3d_realistic.npz'
elif args.src =='t':
    source_path = t
    npz_path_source = '/home/rpu2/scratch/code/MPI3D_data/mpi3d_toy.npz'

if args.tgt =='rl':
    target_path = rl
    npz_path_target = '/home/rpu2/scratch/code/MPI3D_data/real.npz'
elif args.tgt =='rc':
    target_path = rc
    npz_path_target = '/home/rpu2/scratch/code/MPI3D_data/mpi3d_realistic.npz'
elif args.tgt =='t':
    target_path = t
    npz_path_target = '/home/rpu2/scratch/code/MPI3D_data/mpi3d_toy.npz'

if args.tgt =='rl':
    target_path_t = rl_t
    npz_path_test = '/home/rpu2/scratch/code/MPI3D_data/real.npz'
elif args.tgt =='rc':
    target_path_t = rc_t
    npz_path_test = '/home/rpu2/scratch/code/MPI3D_data/mpi3d_realistic.npz'
elif args.tgt =='t':
    target_path_t = t_t
    npz_path_test = '/home/rpu2/scratch/code/MPI3D_data/mpi3d_toy.npz'


store_name = f'source_{args.src}_target_{args.tgt}'

dsets = {"train": ImageList(open(source_path).readlines(), npz_path=npz_path_source, transform=data_transforms["train"]),
         "val": ImageList(open(target_path).readlines(), npz_path=npz_path_target, transform=data_transforms["val"]),
         "test": ImageList(open(target_path_t).readlines(), npz_path=npz_path_test, transform=data_transforms["test"])}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                               shuffle=True, num_workers=8)
                for x in ['train', 'val']}
dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                   shuffle=False, num_workers=16)

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val','test']}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def Regression_test(loader, model):
    MSE = [0, 0, 0]
    MAE = [0, 0, 0]
    number = 0
    upper , bottom, gt_upper, gt_bottom = [], [], [], []
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(loader['test']):
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels1 = labels[:, 0]
            labels2 = labels[:, 1]
            labels1 = labels1.unsqueeze(1)
            labels2 = labels2.unsqueeze(1)
            labels = torch.cat((labels1, labels2), dim=1)
            labels = labels.float() / 39
            pred = model(imgs)
            MSE[0] += torch.nn.MSELoss(reduction='sum')(pred[:, 0], labels[:, 0])
            MAE[0] += torch.nn.L1Loss(reduction='sum')(pred[:, 0], labels[:, 0])
            MSE[1] += torch.nn.MSELoss(reduction='sum')(pred[:, 1], labels[:, 1])
            MAE[1] += torch.nn.L1Loss(reduction='sum')(pred[:, 1], labels[:, 1])
            MSE[2] += torch.nn.MSELoss(reduction='sum')(pred, labels)
            MAE[2] += torch.nn.L1Loss(reduction='sum')(pred, labels)
            number += imgs.size(0)
            bottom.extend(pred[:, 0].cpu().tolist())
            upper.extend(pred[:, 1].cpu().tolist())
            gt_bottom.extend(labels[:, 0].cpu().tolist())
            gt_upper.extend(labels[:, 1].cpu().tolist())
    for j in range(3):
        MSE[j] = MSE[j] / number
        MAE[j] = MAE[j] / number
    print("\tMSE : {0},{1}\n".format(MSE[0], MSE[1]))
    print("\tMAE : {0},{1}\n".format(MAE[0], MAE[1]))
    print("\tMSEall : {0}\n".format(MSE[2]))
    print("\tMAEall : {0}\n".format(MAE[2]))
    x = [i for i in range(len(upper))]
    plt.plot(x, gt_upper, label='labels')
    plt.plot(x, upper, label='upper_preds')
    plt.legend()
    plt.draw()
    plt.savefig('./pic-{}.png'.format(upper))
    plt.plot(x, gt_bottom, label='labels')
    plt.plot(x, bottom, label='bottom_preds')
    plt.legend()
    plt.draw()
    plt.savefig('./pic-{}.png'.format(bottom))


class Model_Regression(nn.Module):
    def __init__(self):
        super(Model_Regression,self).__init__()
        self.model_fc = model.Resnet18Fc()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.classifier_layer = nn.Sequential(self.classifier_layer,  nn.Sigmoid())
        self.predict_layer = nn.Sequential(self.model_fc,self.classifier_layer)
    def forward(self,x):
        feature = self.model_fc(x)
        outC= self.classifier_layer(feature)
        return(outC,feature)



Model_R = Model_Regression()
Model_R = Model_R.to(device)

Model_R.train(True)
criterion = {"regressor": nn.MSELoss()}
optimizer_dict = [{"params": filter(lambda p: p.requires_grad, Model_R.model_fc.parameters()), "lr": 0.1},
                  {"params": filter(lambda p: p.requires_grad, Model_R.classifier_layer.parameters()), "lr": 1}]
optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
train_cross_loss = train_rsd_loss = train_total_loss = 0.0
len_source = len(dset_loaders["train"]) - 1
len_target = len(dset_loaders["val"]) - 1
param_lr = []
iter_source = iter(dset_loaders["train"])
iter_target = iter(dset_loaders["val"])

Model_R.eval()
Regression_test(dset_loaders, Model_R.predict_layer)



