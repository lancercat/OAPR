import torch
import torch.nn.functional as trnf
from torch import nn


class spatial_attention_rect(nn.Module):

    def __init__(this,w_parts=1,h_parts=1,detached=False):
        super(spatial_attention_rect, this).__init__();
        this.r=h_parts;
        this.c=w_parts;


    def forward(this, input):
        d=input[-1];
        x=torch.diag(torch.ones(this.r*this.c,dtype=d.dtype,device=d.device)).reshape(1,this.r*this.c,this.r,this.c);
        if(x.shape[-1]!=d.shape[-1]):
            x=trnf.interpolate(x,[d.shape[-2],d.shape[-1]],mode="area");
        return x;
