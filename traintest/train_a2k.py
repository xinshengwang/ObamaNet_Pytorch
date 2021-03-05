from utils.config import cfg
from sklearn.preprocessing import MinMaxScaler
import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.util import *
import pdb

def train_a2k(model,train_loader,val_loader,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    loss_meter = AverageMeter()
    best_epoch, best_val_loss = 0, np.inf
    global_step, epoch = 0, 0
    progress = []
    start_time = time.time()
    exp_dir = args.exp_dir
    save_model_dir = os.path.join(exp_dir,'models/audio2kps')
    
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    
    def _save_progress():
        progress.append([epoch,global_step,best_epoch,best_val_loss,time.time()-start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if args.resume:
        progress_pkl = "%s/progress.pkl" % exp_dir
        epoch,global_step,best_epoch,best_val_loss = load_progress(progress_pkl)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_val_loss = %.4f" % best_val_loss)
        model.load_state_dict(torch.load(os.path.join(save_model_dir,'best_a2k.pth')))

    if not isinstance(model,torch.nn.DataParallel):
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), args.learning_rate,
                weight_decay=args.weight_decay, betas=(0.95, 0.999))
        
    criterion = nn.MSELoss()

    while epoch<=args.max_epochs:
        adjust_learning_rate(args.learning_rate,args.lr_decay,optimizer,epoch,args)
        model.train()
        for i, (audio,keyp,mask) in enumerate(train_loader):
            global_step += 1
            audio = audio.float().to(device)
            keyp = keyp.float().to(device)
            mask = mask.long().to(device)
            optimizer.zero_grad()
            pred = model(audio)
            loss = criterion(pred*mask,keyp)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(),audio.shape[0])
            if global_step % args.display == 0: 
                info = 'Epoch:[{0}] Global_step:[{1}] Average Loss:{loss_meter.avg:.4f} Loss:{loss_meter.val:.4f}'.format(epoch,global_step,loss_meter=loss_meter)
                print(info)
        val_loss = eval(model,val_loader,args,epoch,criterion)
        if val_loss< best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(),os.path.join(save_model_dir,'best_a2k.pth'))
        info = 'Epoch:[{0}] Global_step:[{1}] Average Loss:{loss_meter.avg:.4f} Val Loss:{val_loss:.4f} \n'.format(epoch,global_step,loss_meter=loss_meter,val_loss=val_loss)
        write_path = os.path.join(exp_dir,'results.text')
        with open(write_path,'a') as f:
            f.writelines(info)
        _save_progress()
        epoch += 1

def eval(model,data_loader,args,epoch,criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_meter = AverageMeter()
    for i, (audio,keyp,mask) in enumerate(data_loader):
        audio = audio.float().to(device)
        keyp = keyp.float().to(device)
        mask = mask.long().to(device)
        with torch.no_grad():
            pred = model(audio)
            val_loss = criterion(pred*mask,keyp)
        val_loss_meter.update(val_loss.item(),audio.shape[0])
    val_loss = val_loss_meter.avg
    save_dir = os.path.join(args.exp_dir,'images',str(epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # pdb.set_trace()
    visualization_lip(pred.reshape(-1,8).cpu().detach().numpy(),keyp.reshape(-1,8).cpu().detach().numpy(),save_dir)
    return val_loss

def _test(model,data_loader,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    save_model_dir = os.path.join(args.exp_dir,'models/audio2kps')
    if not isinstance(model,torch.nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)

    if args.resume:
        progress_pkl = "%s/progress.pkl" % args.exp_dir
        prog, epoch, global_step, best_epoch, best_val_loss = load_progress(progress_pkl)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_val_loss = %.4f" % best_val_loss)
        model.load_state_dict(torch.load(os.path.join(save_model_dir,'best_a2k.pth')))
    
    model.eval()

    test_loss_meter = AverageMeter()
    for i, (audio,keyp,key) in enumerate(data_loader):
        # pdb.set_trace()
        audio = audio.float().to(device).squeeze(0)
        keyp = keyp.float().to(device).squeeze(0)
        save_dir = os.path.join(args.exp_dir,'test_gt_d5',key[0])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with torch.no_grad():
            pred = model(audio)
            visualization_lip(pred.reshape(-1,8).cpu().detach().numpy(),keyp.reshape(-1,8).cpu().detach().numpy(),save_dir)
        print('save the video of %s'%(key[0]))
            # val_loss = criterion(pred*mask,keyp)
        # test_loss_meter.update(val_loss.item(),audio.shape[0])
    # test_loss = test_loss_meter.avg
    # return test_loss

