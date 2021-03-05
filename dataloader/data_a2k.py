import os
import pickle as pkl
import json
from utils.config import cfg
import numpy as np 
import torch
import torch.utils.data as data
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
#########################################################################################

#########################################################################################

# Load the files

def load_split(split,root):
    if split == 'train':
        path = os.path.join(root,'split/train.json')
    elif split == 'test':
        path = os.path.join(root,'split/test.json')
    else:
        path = os.path.join(root,'split/val.json')
    with open(path) as f:
        keys = json.load(f)
    return keys

def delayArray(arr,delay, dummy_var=0):
    arr[:,delay:, :] = arr[:,:(arr.shape[1] - delay), :]
    arr[:,:delay, :] = dummy_var
    return arr


def load_and_processing(args,split):
    data_root = args.path_root
    audio_kp_path = os.path.join(data_root,'for_ObamaNet/audio_kp/audio_kp_mel.pickle')
    video_kp_path = os.path.join(data_root,'for_ObamaNet/pca/pkp.pickle')
    pca_path = os.path.join(data_root,'for_ObamaNet/pca/pca.pickle') 
    
    with open(audio_kp_path, 'rb') as pkl_file:
        audio_kp = pkl.load(pkl_file)
    with open(video_kp_path, 'rb') as pkl_file:
        video_kp = pkl.load(pkl_file)
    with open(pca_path, 'rb') as pkl_file:
        pca = pkl.load(pkl_file)
    
    split_keys = load_split(split,data_root)

    # Get the data

    X, y = [], [] # Create the empty lists
    # Get the common keys
    keys_audio = audio_kp.keys()
    keys_video = video_kp.keys()
    av_keys = sorted(list(set(keys_audio).intersection(set(keys_video))))
    keys = sorted(list(set(av_keys).intersection(set(split_keys))))

    time_delay = cfg.time_delay
    image_ids = []
    i = 0
    for key in keys:
        image_id = os.path.join(key,str(i))
        image_ids.append(image_id)
        audio = audio_kp[key]
        video = video_kp[key]
        if (len(audio) > len(video)):
            audio = audio[0:len(video)]
        else:
            video = video[0:len(audio)]
        audio = np.array(audio)
        if i == 0:
            X = audio
            y = video
        else:
            X = np.vstack((X,audio))
            y = np.vstack((y,video))
        i += 1

    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalery = MinMaxScaler(feature_range=(0, 1))

    X = scalerX.fit_transform(X)
    y = scalery.fit_transform(y)

    total_len = X.shape[0]
    total_items = (total_len // args.time_steps) * args.time_steps
    X = X[:total_items]
    y = y[:total_items]
    X = X.reshape(-1,args.time_steps,X.shape[-1])    
    y = y.reshape(-1,args.time_steps,y.shape[-1])  
    y = delayArray(y,args.time_delay)
    mask = np.ones((y.shape[0],args.time_steps,1))
    mask = delayArray(mask,args.time_delay)
    
    return X, y, mask, image_ids

def load_test_data(key,args):
    data_root = args.path_root
    audio_kp_path = os.path.join(data_root,'for_ObamaNet/audio_kp/audio_kp_mel.pickle')
    video_kp_path = os.path.join(data_root,'for_ObamaNet/pca/pkp.pickle')
    pca_path = os.path.join(data_root,'for_ObamaNet/pca/pca.pickle') 
    
    with open(audio_kp_path, 'rb') as pkl_file:
        audio_kp = pkl.load(pkl_file)
    with open(video_kp_path, 'rb') as pkl_file:
        video_kp = pkl.load(pkl_file)
    with open(pca_path, 'rb') as pkl_file:
        pca = pkl.load(pkl_file)
 
    # Get the data

    time_delay = cfg.time_delay

    audio = audio_kp[key]
    video = video_kp[key]
    # if (len(audio) > len(video)):
    #     audio = audio[0:len(video)]
    # else:
    #     video = video[0:len(audio)]
    audio = np.array(audio)
    X = audio
    y = video


    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalery = MinMaxScaler(feature_range=(0, 1))

    X = scalerX.fit_transform(X)
    y = scalery.fit_transform(y)

    total_len = X.shape[0]
    X = X.reshape(1,-1,X.shape[-1])    
    y = y.reshape(1,-1,y.shape[-1])  
    y = delayArray(y,args.time_delay)
    mask = np.ones((y.shape[0],args.time_steps,1))
    mask = delayArray(mask,args.time_delay)
    image_ids = np.arange(X.shape[0])
    return X, y




    """
    split1 = int(0.8*X.shape[0])
    split2 = int(0.9*X.shape[0])

    # train_X = X[0:split1]
    # train_y = y[0:split1]
    # val_X = X[split1:split2]
    # val_y = y[split1:split2]
    # test_X = X[split2:]
    # test_y = y[split2:]
    if split =='train':
        return X[0:split1], y[0:split1], mask[0:split1]
    elif split == 'test':
        return X[split2:], y[split2:], mask[split2:]
    else:
        return X[split1:split2], y[split1:split2], mask[split1:split2]
    """

class a2k_data(data.Dataset):
    def __init__(self,args,split='train'):
        self.split = split
        self.args = args
        if split == 'test':
            self.filenames = load_split(split,args.path_root)
        else:
            self.audio, self.keyps, self.mask, self.image_ids = load_and_processing(self.args,self.split)
        
    def __getitem__(self,index):
        if self.split == 'test':
            filename = self.filenames[index]
            audios, keyps = load_test_data(filename,self.args)
            return audios, keyps, filename
        else:
            audio = self.audio[index]
            keyps = self.keyps[index]
            mask = self.mask[index]
            return audio, keyps, mask
    def __len__(self):
        if self.split == 'test':
            return len(self.filenames)
        else:
            return self.audio.shape[0]
        