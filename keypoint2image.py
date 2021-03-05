import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
import model.Unet as net
from dataloader.data_k2i import k2i_data
from traintest.train_k2i import train_k2i, test

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def set_seed(args):
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed(args.manualSeed)  
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(args.manualSeed + worker_id)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/avatar/Obama/Obama/clip',
                        help='path to trn dataset')
    parser.add_argument('--testroot',default='output/aud2kyp/test')
    parser.add_argument('--train',default=True,type=bool)
    parser.add_argument('--exp_dir', default='output/kyp2img', 
                        help='path to save')
    parser.add_argument('--run_ID',default='05_L100')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--manualSeed',type=int,default=200)
    parser.add_argument('--inputChannelSize', type=int, default=3, help='size of the input channels')
    parser.add_argument('--outputChannelSize', type=int,default=3, help='size of the output channels')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
    parser.add_argument('--lambdaGAN', type=float, default=1, help='lambdaGAN')
    parser.add_argument('--lambdaIMG', type=float, default=1, help='lambdaIMG')
    parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--UnetD',default='',help='path to UnetD (Unconditional discriminator)')
    parser.add_argument('--Uncond_loss',action='store_true',default=False) #action='store_true'
    parser.add_argument('--only_L1',action='store_true',default=False)
    parser.add_argument('--max_epochs',default=100)
    parser.add_argument('--lrd_start',default=30)
    parser.add_argument('--lr_decay',type=int,default=120,help='after lr_decay epochs the learning rate decreases to zero')
    parser.add_argument('--display', type=int, default=100, help='interval for displaying train-logs')
    parser.add_argument('--evalIter', type=int, default=1000, help='interval for evauating(generating) images from valDataroot')
    parser.add_argument('--alternative_G',action='store_true',default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args)
    if args.train:
        dataset_train = k2i_data(args.dataroot,split='train')
        dataset_val = k2i_data(args.testroot,split='test')

        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size,
            drop_last=True, shuffle=True,num_workers=args.workers,worker_init_fn=worker_init_fn)
        val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.batch_size,
            drop_last=False, shuffle=False,num_workers=args.workers,worker_init_fn=worker_init_fn)
   
        train_k2i(net,train_loader,val_loader,args)
    else:
        dataset_test = k2i_data(args.testroot,split='test')
        test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.batch_size,
            drop_last=False, shuffle=False,num_workers=args.workers,worker_init_fn=worker_init_fn)
        test(net,test_loader,args)