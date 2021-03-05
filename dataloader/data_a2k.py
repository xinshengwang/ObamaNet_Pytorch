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
    time_delay = cfg.look_back
    i = 0
    for key in keys:
        audio = audio_kp[key]
        video = video_kp[key]

        if (len(audio) > len(video)):
            audio = audio[0:len(video)]
        else:
            video = video[0:len(audio)]
        start = (time_delay-look_back) if (time_delay-look_back > 0) else 0
        for i in range(start, len(audio)-look_back):
            a = np.array(audio[i:i+look_back])
            v = np.array(video[i+look_back-time_delay]).reshape((1, -1))
            X.append(a)
            y.append(v)

    X = np.array(X)
    y = np.array(y)
    shapeX = X.shape
    shapey = y.shape
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(-1, y.shape[2])

    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalery = MinMaxScaler(feature_range=(0, 1))

    X = scalerX.fit_transform(X)
    y = scalery.fit_transform(y)

    X = X.reshape(shapeX)
    y = y.reshape(shapey[0], shapey[2])
    
    return X, y

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
    look_back = cfg.look_back

    audio = audio_kp[key]
    video = video_kp[key]
    audio = audio_kp[key]
        video = video_kp[key]

    if (len(audio) > len(video)):
        audio = audio[0:len(video)]
    else:
        video = video[0:len(audio)]
    start = (time_delay-look_back) if (time_delay-look_back > 0) else 0
    for i in range(start, len(audio)-look_back):
        a = np.array(audio[i:i+look_back])
        v = np.array(video[i+look_back-time_delay]).reshape((1, -1))
        X.append(a)
        y.append(v)

    X = np.array(X)
    y = np.array(y)
    shapeX = X.shape
    shapey = y.shape
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(-1, y.shape[2])

    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalery = MinMaxScaler(feature_range=(0, 1))

    X = scalerX.fit_transform(X)
    y = scalery.fit_transform(y)

    X = X.reshape(shapeX)
    y = y.reshape(shapey[0], shapey[2])
    
    return X, y


class a2k_data(data.Dataset):
    def __init__(self,args,split='train'):
        self.split = split
        self.args = args
        if split == 'test':
            self.filenames = load_split(split,args.path_root)
        else:
            self.audio, self.keyps = load_and_processing(self.args,self.split)
        
    def __getitem__(self,index):
        if self.split == 'test':
            filename = self.filenames[index]
            audios, keyps = load_test_data(filename,self.args)
            return audios, keyps, filename
        else:
            audio = self.audio[index]
            keyps = self.keyps[index]
            return audio, keyps
    def __len__(self):
        if self.split == 'test':
            return len(self.filenames)
        else:
            return self.audio.shape[0]
        