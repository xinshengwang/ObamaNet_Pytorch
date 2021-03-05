from utils.util import *
from utils.config import cfg
import torch.optim as optim
import torch.nn as nn
import cv2
import time
import os
from torch.autograd import Variable
import pdb

def train_k2i(model,train_loader,val_loader,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    loss_meter = AverageMeter()
    best_epoch, best_SSIM = 0, -np.inf
    ganIterations, epoch = 0, 0
    progress = []
    start_time = time.time()
    exp_dir = os.path.join(args.exp_dir,args.run_ID)
    save_model_dir = os.path.join(exp_dir,'models')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    if not args.alternative_G:
        netG = model.G(cfg.Unet.inputChannelSize, cfg.Unet.outputChannelSize, cfg.Unet.ngf)
    else:
        netG = model.UnetGenerator(cfg.Unet.inputChannelSize, cfg.Unet.outputChannelSize, cfg.Unet.ngf)
    
    netG.apply(weights_init)
    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG))
    netD = model.D(cfg.Unet.inputChannelSize + cfg.Unet.outputChannelSize, cfg.Unet.ndf)
    netD.apply(weights_init)
    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD))
    
    if args.Uncond_loss:
        UnetD = model.D(cfg.Unet.inputChannelSize, cfg.Unet.ndf)
        UnetD.apply(weights_init)
        if args.UnetD != '':
            UnetD.load_state_dict(torch.load(args.UnetD))

    criterionBCE = nn.BCELoss()
    criterionMSE = nn.MSELoss()
    criterionCAE = nn.L1Loss()

    sizePatchGAN = 30

    # image pool storing previously generated samples from G
    imagePool = ImagePool(cfg.Unet.poolSize)
    # NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper)
    lambdaGAN = args.lambdaGAN
    lambdaIMG = args.lambdaIMG

    netD.to(device)
    netG.to(device)
    criterionBCE.to(device)
    criterionMSE.to(device)
    criterionCAE.to(device)
    if args.Uncond_loss:
        UnetD.to(device)


    # get optimizer
    optimizerD = optim.Adam(netD.parameters(), lr = args.lrD, betas = (args.beta1, 0.999), weight_decay=args.wd) # 
    optimizerG = optim.Adam(netG.parameters(), lr = args.lrG, betas = (args.beta1, 0.999), weight_decay=0.0 ) # 
    if args.Uncond_loss:
        optimizerUD = optim.Adam(UnetD.parameters(), lr = args.lrD, betas = (args.beta1, 0.999)) 

    real_labels = Variable(torch.FloatTensor(args.batch_size,1,sizePatchGAN,sizePatchGAN).fill_(1)).to(device)
    fake_labels = Variable(torch.FloatTensor(args.batch_size,1,sizePatchGAN,sizePatchGAN).fill_(0)).to(device)

    while epoch<=args.max_epochs:
        adjust_learning_rate(args.lrD,args.lr_decay,optimizerD,epoch,args)
        adjust_learning_rate(args.lrG,args.lr_decay,optimizerG,epoch,args)
        netG.train()
        netD.train()
        if args.Uncond_loss:
            adjust_learning_rate(args.lrD,args.lr_decay,optimizerUD,epoch,args)
            UnetD.train()
            
        for i, (input,target) in enumerate(train_loader):
            input = input.float().to(device)
            target = target.float().to(device)
            output = netG(input)
            if not args.only_L1:
                # update discriminator
                for p in netD.parameters(): 
                    p.requires_grad = True
                optimizerD.zero_grad()
                fake = torch.cat([output, input], 1)
                pred_fake = netD(fake.detach())
                loss_D_fake = criterionMSE(pred_fake,fake_labels)

                real = torch.cat([target, input], 1)
                pred_real = netD(real)
                loss_D_real = criterionMSE(pred_real,real_labels)

                loss_D = (loss_D_fake + loss_D_real) * 0.5

                loss_D.backward()
                optimizerD.step()
                for p in netD.parameters(): 
                    p.requires_grad = False
            
                # update unconditional discriminator        
                if args.Uncond_loss:
                    for p in UnetD.parameters(): 
                        p.requires_grad = True
                    optimizerUD.zero_grad()
                    pred_fake_U = UnetD(output.detach())
                    pred_real_U = UnetD(target)
                    loss_UD = (criterionMSE(pred_fake_U,fake_labels) + criterionMSE(pred_real_U,real_labels)) * 0.5
                    loss_UD.backward()
                    optimizerUD.step()
                    for p in UnetD.parameters(): 
                        p.requires_grad = False

                # update generator
                optimizerG.zero_grad()
                pred_fake = netD(fake)
                loss_G_GAN = criterionMSE(pred_fake,real_labels)
                loss_G_L1 = criterionCAE(target,output)*lambdaIMG
                loss_G = loss_G_GAN + loss_G_L1
                
                
                if args.Uncond_loss:
                    pred_fake_U = UnetD(output)
                    loss_G_Ucondi = criterionMSE(pred_fake_U,real_labels)
                    loss_G += loss_G_Ucondi
            else:
                loss_G = criterionCAE(target,output)

            loss_G.backward()
            optimizerG.step()           

            ganIterations += 1

            if ganIterations % args.display == 0:
                save_info = os.path.join(exp_dir,'train_info_k2i.text')
                if args.only_L1:
                    info = 'epoch: {0} itr: {1} loss_G: {lossG:.4f} \n'.format(epoch,ganIterations,lossG=loss_G.item())
                elif args.Uncond_loss:
                    info = 'epoch: {0} itr: {1} loss_D: {lossD:.4f} loss_UD: {lossUD:.4f} loss_G: {lossG:.4f} \n'.format(epoch,ganIterations,lossD=loss_D.item(),lossUD=loss_UD.item(),lossG=loss_G.item()) #
                else:
                    info = 'epoch: {0} itr: {1} real_D: {realD:.4f} fake_D: {fakeD:.4f} G_adv: {lossG1:.4f} G_L1:{lossG2:.4f}\n'.format(epoch,ganIterations,realD=loss_D_real.item(),fakeD=loss_D_fake.item(),lossG1=loss_G_GAN.item(),lossG2=loss_G_L1.item()) #
                print(info)
                with open(save_info,'a') as f:
                    f.writelines(info)
            if ganIterations % args.evalIter == 0:
                eval(netG,val_loader,args,ganIterations)       
                torch.save(netG.state_dict(), '%s/netG_itr_%d.pth' % (save_model_dir, ganIterations))
                netG.train()
                # if not args.only_L1:
                #     torch.save(netD.state_dict(), '%s/netD_itr_%d.pth' % (save_model_dir, ganIterations)) 
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (save_model_dir, epoch))
        # if not args.only_L1:
        #     torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (save_model_dir, epoch))
        epoch += 1

def eval(netG,data_loader,args,ganIterations):
    netG.eval()
    for i, (input,img_ID) in enumerate(data_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = input.float().to(device)
        output = netG(input)
        save_root = os.path.join(args.exp_dir,args.run_ID,'images',str(ganIterations))
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_path = os.path.join(save_root,img_ID[0].replace('/','_') + '.png')
        save_image(output,save_path,normalize=True)

def _eval(netG,data_loader,args,ganIterations):
    netG.eval()
    for i, (input,img_ID) in enumerate(data_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = input.float().to(device)
        output = netG(input)
        
        for j in range(len(img_ID)):
            imgid = img_ID[j]
            save_root = os.path.join(args.exp_dir,args.run_ID,'test_val',str(ganIterations),imgid.split('/')[0])
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            save_path = os.path.join(save_root,imgid.split('/')[-1]) + '.png'
            save_image_single(output[j],save_path)

def test(model,data_loader,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(args.exp_dir,args.run_ID)
    save_image_dir = os.path.join(exp_dir,'test')
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)

    if not args.alternative_G:
        netG = model.G(cfg.Unet.inputChannelSize, cfg.Unet.outputChannelSize, cfg.Unet.ngf)
    else:
        netG = model.UnetGenerator(cfg.Unet.inputChannelSize, cfg.Unet.outputChannelSize, cfg.Unet.ngf)
    netG.load_state_dict(torch.load(args.netG))
    netG.to(device)
    for i, (input,img_ID) in enumerate(data_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input = input.float().to(device)
        # pdb.set_trace()
        output = netG(input)
        
        # pdb.set_trace()
        for j in range(len(img_ID)):
            imgid = img_ID[j]
            save_root = os.path.join(args.exp_dir,args.run_ID,'test',imgid.split('/')[0])
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            save_path = os.path.join(save_root,imgid.split('/')[-1]) + '.png'
            save_image_single(output[j],save_path)
        print(i)
