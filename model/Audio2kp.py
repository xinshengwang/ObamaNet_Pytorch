import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import cfg

"""
Audio-to-keypoint model
"""
class audio2kp(nn.Module):
    def __init__(self):
        super(audio2kp,self).__init__()
        self.lstm = nn.LSTM(cfg.in_channel,cfg.hidden_size,batch_first=True,dropout=cfg.drop)
        self.fc = nn.Linear(cfg.hidden_size,cfg.out_channel)
    
    def forward(self,x):
        x,_ = self.lstm(x)
        x = self.fc(x)
        return x