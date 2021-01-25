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

class db_deco_easy(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(db_deco_easy, self).__init__()
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
        self.outc = OutConv(64, n_classes+1)

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
        rebu_mask = self.outc(x)   # [1,   512, 512]
        return (rebu_mask[:,0:1],rebu_mask[:,1:2])


class db_deco(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(db_deco, self).__init__()
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

        self.up21 = Up(1024, 512 // factor, bilinear)
        self.up22 = Up(512, 256 // factor, bilinear)
        self.up23 = Up(256, 128 // factor, bilinear)
        self.up24 = Up(128, 64, bilinear)
        self.outc2 = OutConv(64, n_classes)

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

        y = self.up21(x5, x4)    # [256, 64,  64]
        y = self.up22(y, x3)     # [128, 128, 128]
        y = self.up23(y, x2)     # [64,  256, 256]
        y = self.up24(y, x1)     # [64,  512, 512]
        rebuild = self.outc2(y)  # [1,   512, 512]
        return (rebuild,logits)


class db_enco(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(db_enco, self).__init__()
        from model.unet import UNet
        self.unet_mask = UNet(n_channels,n_classes,bilinear)
        self.unet_rebuild = UNet(n_channels,n_classes,bilinear)

    def forward(self, x):
        logits = self.unet_mask(x)              # [1,   512, 512]
        rebuild = self.unet_rebuild(logits)     # [1,   512, 512]
        return (rebuild,logits)

class db_enco_up4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(db_enco_up4, self).__init__()
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

        self.down21 = Down(64, 128)
        self.down22 = Down(128, 256)
        self.down23 = Down(256, 512)
        self.down24 = Down(512, 1024 // factor)
        self.up21 = Up(1024, 512 // factor, bilinear)
        self.up22 = Up(512, 256 // factor, bilinear)
        self.up23 = Up(256, 128 // factor, bilinear)
        self.up24 = Up(128, 64, bilinear)
        self.outc2 = OutConv(64, n_classes)

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

        y2 = self.down21(x)
        y3 = self.down22(y2)
        y4 = self.down23(y3)
        y5 = self.down24(y4)
        y = self.up21(y5, y4)    # [256, 64,  64]
        y = self.up22(y, y3)     # [128, 128, 128]
        y = self.up23(y, y2)     # [64,  256, 256]
        y = self.up24(y, x)      # [64,  512, 512]
        rebuild = self.outc2(y)  # [1,   512, 512]
        return (rebuild,logits)

################################################ NetWork End ############################################################


class loss(weak_loss):
    def __init__(self, wr, wm):
        super(loss, self).__init__()
        self.wr=wr
        self.wm=wm
        self.loss_rebuild = nn.MSELoss()
        # self.loss_mask = nn.BCEWithLogitsLoss()
        self.loss_mask = FocalLoss2(use_logits=True)
    def get_loss(self, pre, tar):
        lrebuild=self.loss_rebuild(pre[0],tar[0])
        lmask=self.loss_mask(pre[1],tar[1])
        l = self.wr*lrebuild+self.wm*lmask
        return l, {'mask':lmask,'rebuild':lrebuild}

class evaluate(weak_evaluate):
    def __init__(self):
        super(evaluate, self).__init__()
        self.cnt=0
        self.act = nn.Sigmoid()

    def get_eval(self, inputs, preds, targets):
        iou_reverse=[]
        iou=[]
        metric=[]
        targets=targets[1]
        preds=preds[1]
        for k in range(inputs.shape[0]):
            p=(self.act(preds[k])<0.5).float()
            t=1-targets
            u=(t+p)>=1
            i=t*p
            iou.append(i.sum()/u.sum())
            pr=1-p
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
            cv2.imshow('t',torch.cat([inputs[i,0],(self.act(preds[i,0])>0.5).float(),targets[i,0]],1).cpu().detach().numpy())
            k=cv2.waitKey(0)
            if k=='q':break
        return k

    def cat_save(self,name,*subim):
        for_save = (torch.cat(subim,1).cpu().detach().numpy()*255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.result_dir,'{}_{}.png'.format(name,self.cnt)),for_save)

    def save(self, inputs, preds, targets, _eval):
        for i in range(inputs.shape[0]):
            self.cat_save('rebuild',preds[0][i,0],targets[0][i,0])
            p,t=(preds[1][i,0]>0).float(),targets[1][i,0]
            tp=(t*p).bool()
            fp=(t*(1-p)).bool()
            fn=((1-t)*p).bool()
            t3=t.unsqueeze(2).repeat(1,1,3)
            im=torch.zeros_like(t3)
            im[tp,:]=1
            im[...,2][fp]=1
            im[...,1][fn]=1
            self.cat_save('mask',im,t3)
            self.cnt+=1

if __name__ == "__main__":
    n=UNet(1,1)
    i=torch.rand(1,1,64,64).float()
    t=torch.randint(0,2,(1,1,64,64)).float()
    o=n(i)
    l=loss()
    print(l(o,t))
    print(o)