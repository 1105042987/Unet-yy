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
        self.imgdir=PJ(cfg['direction'],'{}_img'.format(mode))
        self.labdir=PJ(cfg['direction'],'{}_label'.format(mode))
        self.namelist = os.listdir(self.imgdir)
        self.train=mode=='train'
        self.transforms = transforms
        self.th=cfg['th']
        self.size=len(self.namelist)
        self.ori=cfg['ori']
        self.repeat=cfg['repeat_time']
        self.cell=cfg['cell']

    def __getitem__(self,idx):
        inputs=cv2.imread(PJ(self.imgdir,self.namelist[idx]),0)
        targets=cv2.imread(PJ(self.labdir,self.namelist[idx]),0)
        if self.train:
            crop=self.transforms(np.concatenate((inputs[...,np.newaxis],targets[...,np.newaxis]),axis=-1))
            inputs=crop[0:1].unsqueeze(0)
            targets=(crop[1:2]>self.th).float().unsqueeze(0)
        else:
            h,w=inputs.shape
            H,W=h//self.cell,w//self.cell
            inputs=self.transforms(inputs)
            targets=(self.transforms(targets)>self.th).float()
            inputs=inputs.reshape(H,self.cell,W,self.cell).permute(0,2,1,3).reshape(H*W,self.cell,self.cell).unsqueeze(1)
            targets=targets.reshape(H,self.cell,W,self.cell).permute(0,2,1,3).reshape(H*W,self.cell,self.cell).unsqueeze(1)
        return inputs,targets

    def __len__(self):
        return self.size*self.repeat


def my_collate(batch):
    data = torch.cat([item[0] for item in batch],0)
    target = torch.cat([item[1] for item in batch],0)
    return [data, target]

def dataloader(cfg, mode):
    if mode=='train':
        bs=cfg['batch_size']
        transforms = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(cfg['cell'],scale=(0.5,2)),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.ToTensor(),
        ])
    else:
        bs=cfg['batch_size']*cfg['cell']//cfg['ori']*cfg['cell']//cfg['ori']
        bs=max(bs,1)
        transforms = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
        ])
    dataset = Dataset(cfg,mode,transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=cfg['shuffle'], num_workers=cfg['num_workers'],collate_fn=my_collate)
    return loader

if __name__ == "__main__":
    mode = 'train'
    direction={
        'test':PJ("D:\\Data","ISBI-2012-EM","new_test_set"),
        "train": PJ("D:\\Data","ISBI-2012-EM","new train set")
    }
    cfg= {
        "file_name": "ISBI",
        "num_workers": 1,
        "th":0.6,
        'repeat_time':1,
        "batch_size": 8,
        "cell":256,
        "ori":512,
        "shuffle":True,
        'direction':direction[mode]
    }
    d=iter(dataloader(cfg,mode))
    i,t=next(d)
    for kkk in range(cfg['batch_size']):
        cv2.imshow('1',np.array(t[kkk,0]))
        cv2.waitKey(0)
    print(i.shape,t.shape)
