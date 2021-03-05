import torch.utils.data as data
import json
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def image_loader(path,normalize):
    image = Image.open(path).convert('RGB')
    if normalize is not None:
        image = normalize(image)
    return image

class k2i_data(data.DataLoader):
    def __init__(self,root,split='train',transform=None):
        self.root = root
        self.split = split
        self.filenames = self.load_filenames(root,split)
        if split == 'val':
            self.filenames = self.filenames[:10]
        elif split == 'test':
            self.filenames = self.filenames[:300]
        # elif split == 'test':
        #     new_filenames = []
        #     for item in self.filenames:
        #         vn,im = item.split('/')
        #         new_im = str((int(im) - 1)).zfill(5) 
        #         new_name = os.path.join(vn,new_im)
        #         new_filenames.append(new_name)
        #     self.filenames = new_filenames
    
            
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def load_filenames(self,root,split):
        if split != 'test':
            path = os.path.join(root,'split/for_kp2img/5fps',split + '.json')
            with open(path,'rb') as f:
                data = json.load(f)
        else:
            data = []
            video_names = os.listdir(root)
            for video_name in video_names:
                video_path = os.path.join(root,video_name)
                image_names = os.listdir(video_path)
                for image_name in image_names:
                    filename = os.path.join(video_name,image_name)
                    data.append(filename)
        return data
    
    def __getitem__(self,index):
        image_name = self.filenames[index]
        if self.split != 'test':
            image_path = os.path.join(self.root,'images_crop',image_name + '.jpg')
            lip_path = os.path.join(self.root,'for_ObamaNet/images_lip',image_name + '.png')
            img = image_loader(image_path,self.norm)
            lip = image_loader(lip_path,normalize=self.norm)        
        else:
            data_path = 'output/aud2kyp/real_fake'
            lip_path = os.path.join(data_path,image_name)
            lip = image_loader(lip_path,normalize=self.norm)  
        if self.split == 'train':
            return lip, img
        # elif self.split == 'test':
        #     return lip, image_name
        else:
            return lip, image_name

    def __len__(self):
        return len(self.filenames)

