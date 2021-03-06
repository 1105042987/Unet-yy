from __future__ import absolute_import
import os 
import sys
base = sys.path[0]
sys.path.append(os.path.abspath(os.path.join(base, "..")))
import cv2
import torch
import numpy as np 
import torchvision.transforms as T
from os.path import join as PJ
from docker.abstract_model import weak_SplitPatch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, transforms=None):
        super(Dataset, self).__init__()
        subname='train' if 'train' in cfg['direction'] else 'test'
        self.imgdir=PJ(cfg['direction'],'{}_img'.format(subname))
        self.labdir=PJ(cfg['direction'],'{}_label'.format(subname))
        self.namelist = os.listdir(self.imgdir)
        self.th=cfg['th']
        self.transforms = transforms
        self.size=len(self.namelist)
        self.repeat=cfg['repeat_time']
        
    def __getitem__(self,idx):
        idx%=self.size
        inputs=cv2.imread(PJ(self.imgdir,self.namelist[idx]),0)
        targets=cv2.imread(PJ(self.labdir,self.namelist[idx]),0)
        crop=self.transforms(np.concatenate((inputs[...,np.newaxis],targets[...,np.newaxis]),axis=-1))
        inputs=crop[0:1]
        targets=(crop[1:2]<=self.th).float()
        return inputs,(inputs,targets)

    def __len__(self):
        return self.size*self.repeat


def dataloader(cfg, mode):
    if mode=='train':
        transforms = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.ToTensor(),
        ])
    else:
        transforms = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
        ])
    dataset = Dataset(cfg,mode,transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'], num_workers=cfg['num_workers'])    
    return loader

if __name__ == "__main__":
    mode = 'test'
    direction={
        'test':PJ("D:\\Data","ISBI-2012-EM","new_test_set"),
        "train": PJ("D:\\Data","ISBI-2012-EM","new train set")
    }
    cfg= {
        "file_name": "ISBI",
        "num_workers": 1,
        "th":0.6,
        'repeat_time':10,
        "batch_size": 1,
        "shuffle":True,
        'direction':direction[mode]
    }
    d=iter(dataloader(cfg,mode))
    for kkk in range(10):
        i,t=next(d)
        print(t.sum()/512/512)
    #     cv2.imshow('1',np.array(i[0,0]))
    #     cv2.waitKey(0)
    print(i.shape,t.shape)
