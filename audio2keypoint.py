from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import pickle as pkl
from time import time
import argparse
import torch
import random
from utils.config import cfg, cfg_from_file
from model.Audio2kp import audio2kp
from dataloader.data_a2k import a2k_data
from traintest.train_a2k import train_a2k

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
    parser = argparse.ArgumentParser(description='Train the audio-to-keypoint network')
    parser.add_argument('--train',type=bool,default=True)
    parser.add_argument('--cfg',dest='cfg_file',type=str,default=None,
                        help='optional config file')
    parser.add_argument('--path_root',type=str,default='/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/avatar/Obama/Obama/clip') 
    parser.add_argument('--resume',type=bool,default=True)
    parser.add_argument('--time_delay',type=int,default=20)
    parser.add_argument('--workers',type=int,default=0,
                        help='number of workers for loading data')  
    parser.add_argument('--manualSeed',type=int,default= 200,
                        help='manual seed')
    parser.add_argument('--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 100)')
    # setting training parameter    
    parser.add_argument("--exp_dir",type=str,default='output/aud2kyp/with_lookback')
    parser.add_argument("--optim", type=str, default="adam",
                        help="training optimizer", choices=["sgd", "adam"])

    parser.add_argument('--learning_rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument("--max_epochs",type=int,default=50)
    parser.add_argument('--lrd_start',default=25,type=int, help='after which epoch the learning rate start to decrease')
    parser.add_argument('--lr_decay', default=50, type=int, metavar='LRDECAY',
                        help='after lr_decay epochs the learning rate decreases to zero')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', 
                        help='weight decay (default: 1e-4)')    
    parser.add_argument('--display',default=50,type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    model = audio2kp()
    dataset_train = a2k_data(args,'train')
    dataset_val = a2k_data(args,'val')
    dataset_test = a2k_data(args,'test')
    data_loader_train = torch.utils.data.DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,worker_init_fn=worker_init_fn)
    data_loader_val = torch.utils.data.DataLoader(dataset_val,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,worker_init_fn=worker_init_fn)
    if args.train:
        train_a2k(model,data_loader_train,data_loader_val,args)
    else:
        from traintest.train_a2k import _test
        data_loader_test = torch.utils.data.DataLoader(dataset_test,batch_size=1,shuffle=False,num_workers=args.workers,worker_init_fn=worker_init_fn)
        _test(model,data_loader_test,args)



