from __future__ import absolute_import
import sys
import os 
base = sys.path[0]
sys.path.append(os.path.abspath(os.path.join(base, "..")))
import cv2
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from docker.abstract_model import weak_evaluate, weak_loss
from model.unet_parts import *
from utils.loss.focal_loss import FocalLoss2

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):       # [1,   512, 512]
        x1 = self.inc(x)        # [64,  512, 512]
        x2 = self.down1(x1)     # [128, 256, 256]
        x3 = self.down2(x2)     # [256, 128, 128]
        x4 = self.down3(x3)     # [512, 64,  64]
        x5 = self.down4(x4)     # [1024/512,32,  32]
        x = self.up1(x5, x4)    # [256, 64,  64]
        x = self.up2(x, x3)     # [128, 128, 128]
        x = self.up3(x, x2)     # [64,  256, 256]
        x = self.up4(x, x1)     # [64,  512, 512]
        logits = self.outc(x)   # [1,   512, 512]
        return logits

class loss(weak_loss):
    def __init__(self,alpha,gamma,focal=False):
        super(loss, self).__init__()
        if focal:
            self.loss=FocalLoss2(alpha=alpha,gamma=gamma,use_logits=True)
            self.name='focal'
        else:
            self.loss=nn.BCEWithLogitsLoss()
            self.name='bce'
    def get_loss(self, pre, tar):
        l=self.loss(pre,tar)
        return l, {self.name:l}

class evaluate(weak_evaluate):
    def __init__(self):
        super(evaluate, self).__init__()
        self.cnt=0
        self.act = nn.Sigmoid()

    def get_eval(self, inputs, preds, targets):
        iou_reverse=[]
        iou=[]
        metric=[]
        for k in range(inputs.shape[0]):
            p=(self.act(preds[k,0])<0.5).float()
            t=1-targets
            u=(t+p)>=1
            i=t*p
            iou.append(i.sum()/u.sum())
            pr=(self.act(preds[k,0])>=0.5).float()
            ur=(targets+pr)>=1
            ir=targets*pr
            iou_reverse.append(ir.sum()/ur.sum())
            metric.append([
                (t*p).sum()/t.sum(),            # tp
                ((1-t)*(1-p)).sum()/(1-t).sum(),    # tn        
                (t*(1-p)).sum()/t.sum(),        # fp    
                ((1-t)*p).sum()/(1-t).sum(),       # fn    
            ])
        metric=np.array(metric)
        return {'iou':np.array(iou),'iou_reverse':np.array(iou_reverse),'tp':metric[:,0],'tn':metric[:,1],'fp':metric[:,2],'fn':metric[:,3]}

    def visualize(self, inputs, preds, targets, _eval):
        for i in range(inputs.shape[0]):
            cv2.imshow('t',torch.cat([inputs[i,0],(self.act(preds[i,0])>0.5).float(),targets[i,0]],-1).cpu().detach().numpy())
            k=cv2.waitKey(0)
            if k=='q':break
        return k

    def cat_save(self,name,*subim):
        for_save = (torch.cat(subim,1).cpu().detach().numpy()*255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.result_dir,'{}_{}.jpg'.format(name,self.cnt)),for_save)

    def save(self, inputs, preds, targets, _eval):
        for i in range(inputs.shape[0]):
            p,t=(preds[i,0]>0).float(),targets[i,0]
            tp=(t*p).bool()
            fp=(t*(1-p)).bool()
            fn=((1-t)*p).bool()
            t3=t.unsqueeze(2).repeat(1,1,3)
            im=torch.zeros_like(t3)
            im[tp,:]=1
            im[...,2][fp]=1
            im[...,1][fn]=1
            self.cat_save('mask',inputs[i,0].unsqueeze(2).repeat(1,1,3),im,t3)
            self.cnt+=1

if __name__ == "__main__":
    n=UNet(1,1)
    i=torch.rand(1,1,64,64).float()
    t=torch.randint(0,2,(1,1,64,64)).float()
    o=n(i)
    l=loss()
    print(l(o,t))
    print(o)