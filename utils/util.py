import math
import torch
import cv2
import os
import subprocess
import numpy as np
import pickle as pkl
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

irange = range

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_progress(prog_pkl,quiet=False):
    def _print(msg):
        if not quiet:
            print(msg)
    with open(prog_pkl,'rb') as f:
        prog = pkl.load(f)
        epoch,global_step,best_epoch,best_val_loss,_ = prog[-1]
    
    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_val_loss", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_val_loss

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch,args):
    """Sets the learning rate to the initial LR decayed by 5 every lr_decay epochs"""
    # if args.lr_policy == 
    # lr = base_lr * (0.2 ** (epoch // lr_decay))
    lr = base_lr * (1 - max(0, epoch - args.lrd_start)/float(lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ImagePool:
  def __init__(self, pool_size=50):
    self.pool_size = pool_size
    if pool_size > 0:
      self.num_imgs = 0
      self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image
    if self.num_imgs < self.pool_size:
      self.images.append(image.clone())
      self.num_imgs += 1
      return image
    else:
      if np.random.uniform(0,1) > 0.5:
        random_id = np.random.randint(self.pool_size, size=1)[0]
        tmp = self.images[random_id].clone()
        self.images[random_id] = image.clone()
        return tmp
      else:
        return image
    


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, fp, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp - A filename(string) or file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)

def save_image_single(tensor, fp):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp - A filename(string) or file object
    """
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer   
    def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)
    def norm_range(t, range):
        if range is not None:
            norm_ip(t, range[0], range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))
    norm_range(tensor, range=None)
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=None)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

"""
Fuctions for draw lips
"""
def subsample(y, fps_from = 100.0, fps_to = 20):
	factor = int(np.ceil(fps_from/fps_to))
	# Subsample the points
	new_y = np.zeros((int(y.shape[0]/factor), 20, 2)) #(timesteps, 20) = (500, 20x2)
	for idx in range(new_y.shape[0]):
		if not (idx*factor > y.shape[0]-1):
			# Get into (x, y) format
			new_y[idx, :, 0] = y[idx*factor, 0:20]
			new_y[idx, :, 1] = y[idx*factor, 20:]
		else:
			break
	# print('Subsampled y:', new_y.shape)
	new_y = [np.array(each) for each in new_y.tolist()]
	# print(len(new_y))
	return new_y

def drawLips(keypoints, new_img, c = (255, 255, 255), th = 1, show = False):
	keypoints = np.float32(keypoints)
	for i in range(48, 59):
		cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[59]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[60]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[54]), tuple(keypoints[64]), color=c, thickness=th)
	cv2.line(new_img, tuple(keypoints[67]), tuple(keypoints[60]), color=c, thickness=th)
	for i in range(60, 67):
		cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)

	if (show == True):
		cv2.imshow('lol', new_img)
		cv2.waitKey(10000)

def getOriginalKeypoints(kp_features_mouth, N, tilt, mean):
	# Denormalize the points
	kp_dn = N * kp_features_mouth
	# Add the tilt
	x, y = kp_dn[:, 0], kp_dn[:, 1]
	c, s = np.cos(tilt), np.sin(tilt)
	x_dash, y_dash = x*c + y*s, -x*s + y*c
	kp_tilt = np.hstack((x_dash.reshape((-1,1)), y_dash.reshape((-1,1))))
	# Shift to the mean
	kp = kp_tilt + mean
	return kp

def visualization_lip(y_pred,y_gt,outputFolder):
    with open('/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/avatar/Obama/Obama/clip/for_ObamaNet/target_masked_images/pkp.pickle', 'rb') as pkl_file:
        video_kp = pkl.load(pkl_file)
    with open('/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/avatar/Obama/Obama/clip/for_ObamaNet/pca/pca.pickle', 'rb') as pkl_file:
        pca = pkl.load(pkl_file)
    # Get the original keypoints file of the target video
    with open('/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/avatar/Obama/Obama/clip/for_ObamaNet/target_masked_images/target_raw_kp.pickle', 'rb') as pkl_file:
        kp = pkl.load(pkl_file)[4:]

    video = video_kp[4:]
    y = np.array(video)
    scalery = MinMaxScaler(feature_range=(0, 1))
    y = scalery.fit_transform(y)
    
    y_pred = scalery.inverse_transform(y_pred)
    y_pred = pca.inverse_transform(y_pred)
    y_pred = subsample(y_pred, 100, 20)

    y_gt = scalery.inverse_transform(y_gt)
    y_gt = pca.inverse_transform(y_gt)
    y_gt = subsample(y_gt, 100, 20)


    # Visualization
    # Cut the other stream according to whichever is smaller
    if (len(kp) < len(y_pred)):
        n = len(kp)
        y_pred = y_pred[:n]
        y_gt = y_gt[:n]
    else:
        n = len(y_pred)
        kp = kp[:n]


    for idx, (x,xg,k) in enumerate(zip(y_pred,y_gt, kp)):

        unit_mouth_kp, N, tilt, mean, unit_kp, keypoints = k[0], k[1], k[2], k[3], k[4], k[5]
        keypointsg = keypoints.copy()
        kps = getOriginalKeypoints(x, N, tilt, mean)
        keypoints[48:68] = kps

        kpsg = getOriginalKeypoints(xg, N, tilt, mean)
        keypointsg[48:68] = kpsg

        imgfile = '/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/avatar/Obama/Obama/clip/for_ObamaNet/target_masked_images/' + str(idx+5).zfill(5) + '.png'
        im = cv2.imread(imgfile)
        im2 = im.copy()
        # drawLips(keypoints, im, c = (255, 255, 255), th = 1, show = False)
        drawLips(keypointsg, im2, c = (255, 255, 255), th = 1, show = False)
        # cv2.imwrite(os.path.join(outputFolder, str(idx).zfill(5) + '.png'), im)
        cv2.imwrite(os.path.join(outputFolder, str(idx).zfill(5) + '_gt.png'), im2)
    print('Done writing', n, 'images')
