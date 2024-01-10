import torch
import torch.nn.functional as trnf
from torch import nn

from neko_sdk.neko_spatial_kit.embeddings.neko_emb_intr import neko_add_embint_se;


class spatial_attention_mk3_seintr(nn.Module):
    def set_se_engine(this,se_channel):
        this.se_engine=neko_add_embint_se(16,16,se_channel);
        pass;
    def set_core(this,ifc,n_parts):
        this.core = torch.nn.Sequential(
            torch.nn.Conv2d(
                ifc, ifc, (3, 3), (1, 1), (1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(ifc),
            torch.nn.Conv2d(ifc, n_parts, (1, 1)),
            torch.nn.Sigmoid(),
        );


    def __init__(this,ifc,n_parts=1,se_channel=32,detached=False):
        super(spatial_attention_mk3_seintr, this).__init__();
        this.set_se_engine(se_channel);
        this.set_core(ifc+se_channel,n_parts);
        this.detached=detached;


    def forward(this, input):
        if(this.detached):
            x=input[0].detach();
        else:
            x = input[0];
        d=input[-1];
        if(x.shape[-1]!=d.shape[-1]):
            x=trnf.interpolate(x,[d.shape[-2],d.shape[-1]],mode="area");
        return this.core(this.se_engine(x));
