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

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class loss(weak_loss):
    def __init__(self):
        super(loss, self).__init__()
        self.loss=nn.BCEWithLogitsLoss()
    def get_loss(self, pre, tar):
        l=self.loss(pre,tar)
        return l, {'BCE':l}

class evaluate(weak_evaluate):
    def __init__(self,ori):
        super(evaluate, self).__init__()
        self.cnt=0
        # self.ori=ori
        # self.subcnt=0
        # self.posx=0
        # self.posy=0
        # self.IMG=np.zeros((ori,ori))
        # self.LAB=np.zeros((ori,ori))

    def get_eval(self, inputs, preds, targets):
        return {}

    def visualize(self, inputs, preds, targets, _eval):
        return None

    def save(self, inputs, preds, targets, _eval):
        self.cnt+=1
        cv2.imwrite(os.path.join(self.result_dir,'{}_pre.png'.format(self.cnt)),np.array(preds[0,0,...].cpu()*255))
        cv2.imwrite(os.path.join(self.result_dir,'{}_tar.png'.format(self.cnt)),np.array(targets[0,0,...].cpu()*255))

        # shape=preds.shape[-1]
        # subNum=(self.ori/shape)**2
        # if self.subcnt<subNum:
        #     self.IMG[self.posy:(self.posy+shape),self.posx:(self.posx+shape)]=np.array(preds[0,0,...].cpu()*255)
        #     self.LAB[self.posy:(self.posy+shape),self.posx:(self.posx+shape)]=np.array(targets[0,0,...].cpu()*255)
        #     self.subcnt+=1
        #     if self.posx+shape<self.ori:
        #         self.posx+=shape
        #     else:
        #         self.posx=0
        #         self.posy+=shape
        # else:
        #     cv2.imwrite(os.path.join(self.result_dir,'{}_pre.png'.format(self.cnt)),self.IMG)
        #     cv2.imwrite(os.path.join(self.result_dir,'{}_tar.png'.format(self.cnt)),self.LAB)
        #     self.cnt+=1
        #     self.subcnt=0
        #     self.posx=0
        #     self.posy=0
        #     self.IMG=np.zeros((self.ori,self.ori))
        #     self.LAB=np.zeros((self.ori,self.ori))

        


class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

if __name__ == "__main__":
    n=UNet(1,1)
    i=torch.rand(1,1,64,64).float()
    t=torch.randint(0,2,(1,1,64,64)).float()
    o=n(i)
    l=loss()
    print(l(o,t))
    print(o)