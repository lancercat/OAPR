import torch
import torch.nn as nn
from torch.nn import functional as trnf

from neko_sdk.neko_spatial_kit.embeddings.neko_emb_intr import neko_add_embint_se

'''
Convolutional Alignment Module
'''

# Current version only supports input whose size is a power of 2, such as 32, 64, 128 etc.
# You can adapt it to any input size by changing the padding or stride.
class neko_CAM_stop_seintr(nn.Module):
    def arm_pred_stop(this,ich_stop,maxT):
        this.lenpred= nn.Linear(ich_stop,maxT,False);
    def arm_last_deconv(this,deconvs,num_channels, nmasks,deconv_ksize,stride):
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, nmasks,
                                                        tuple(deconv_ksize),
                                                        tuple(stride),
                                                        (int(deconv_ksize[0] / 4.), int(deconv_ksize[1] / 4.))),
                                     nn.Sigmoid()));
        return deconvs;
    def arm_semodule(this, scales, nmasks, depth,maxT, num_channels,num_se_channels):
        this.semods=[];
        if(type(num_se_channels) is int):
            for cid in range(len(scales)):
                this.semods.append(neko_add_embint_se(scales[cid][1],scales[cid][2],num_se_channels));
                this.add_module("se_"+str(cid),this.semods[-1]);
        else:
            print("error");
            exit(9);

    def arm_stem(this, scales, nmasks, depth,maxT, num_channels,num_se_channels):
        # cascade multiscale features
        this.arm_semodule( scales, nmasks, depth,maxT, num_channels,num_se_channels)
        fpn = []
        for i in range(1, len(scales)):
            assert not (scales[i-1][1] / scales[i][1]) % 1, 'layers scale error, from {} to {}'.format(i-1, i)
            assert not (scales[i-1][2] / scales[i][2]) % 1, 'layers scale error, from {} to {}'.format(i-1, i)
            ksize = [3,3,5] # if downsampling ratio >= 3, the kernel size is 5, else 3
            r_h, r_w = int(scales[i-1][1] / scales[i][1]), int(scales[i-1][2] / scales[i][2])
            ksize_h = 1 if scales[i-1][1] == 1 else ksize[r_h-1]
            ksize_w = 1 if scales[i-1][2] == 1 else ksize[r_w-1]
            fpn.append(nn.Sequential(nn.Conv2d(scales[i-1][0], scales[i][0],
                                              (ksize_h, ksize_w),
                                              (r_h, r_w),
                                              (int((ksize_h - 1)/2), int((ksize_w - 1)/2))),
                                     nn.BatchNorm2d(scales[i][0]),
                                     nn.ReLU(True)))
        this.fpn = nn.Sequential(*fpn)
        # convolutional alignment
        # convs
        assert depth % 2 == 0, 'the depth of CAM must be a even number.'
        in_shape = scales[-1]
        strides = []
        conv_ksizes = []
        deconv_ksizes = []
        h, w = in_shape[1], in_shape[2]
        for i in range(0, int(depth / 2)):
            stride = [2] if 2 ** (depth/2 - i) <= h else [1]
            stride = stride + [2] if 2 ** (depth/2 - i) <= w else stride + [1]
            strides.append(stride)
            conv_ksizes.append([3, 3])
            deconv_ksizes.append([_ ** 2 for _ in stride])
        convs = [nn.Sequential(nn.Conv2d(in_shape[0], num_channels,
                                        tuple(conv_ksizes[0]),
                                        tuple(strides[0]),
                                        (int((conv_ksizes[0][0] - 1)/2), int((conv_ksizes[0][1] - 1)/2))),
                               nn.BatchNorm2d(num_channels),
                               nn.ReLU(True))]
        for i in range(1, int(depth / 2)):
            convs.append(nn.Sequential(nn.Conv2d(num_channels, num_channels,
                                                tuple(conv_ksizes[i]),
                                                tuple(strides[i]),
                                                (int((conv_ksizes[i][0] - 1)/2), int((conv_ksizes[i][1] - 1)/2))),
                                       nn.BatchNorm2d(num_channels),
                                       nn.ReLU(True)))
        this.convs = nn.Sequential(*convs)
        # deconvs
        ich_stop=num_channels;

        deconvs = []
        for i in range(1, int(depth / 2)):
            deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels,
                                                           tuple(deconv_ksizes[int(depth/2)-i]),
                                                           tuple(strides[int(depth/2)-i]),
                                                           (int(deconv_ksizes[int(depth/2)-i][0]/4.), int(deconv_ksizes[int(depth/2)-i][1]/4.))),
                                         nn.BatchNorm2d(num_channels),
                                         nn.ReLU(True)))
            ich_stop += num_channels;
        this.arm_pred_stop(ich_stop,maxT);
        deconvs =this.arm_last_deconv(deconvs,num_channels, nmasks,deconv_ksizes[0],strides[0]);
        this.deconvs = nn.Sequential(*deconvs)

    def __init__(this, scales, maxT, depth, num_channels,num_se_channels,detached=False):
        super(neko_CAM_stop_seintr, this).__init__();
        this.detached=detached;
        this.arm_stem(scales,maxT,depth,maxT,num_channels,num_se_channels)

    def make_att(this, x, t):
        return this.deconvs[-1](x)
    def att_len_core(this,input,mask,t_override):
        x = input[0]
        for i in range(0, len(this.fpn)):
            x = this.fpn[i](x) + input[i + 1]
        conv_feats = []
        stoppers = []
        for i in range(0, len(this.convs)):
            x = this.convs[i](x)
            conv_feats.append(x)
        stoppers.append(x.mean(-1).mean(-1));
        for i in range(0, len(this.deconvs) - 1):
            x = this.deconvs[i](x)
            f = conv_feats[len(conv_feats) - 2 - i]
            stoppers.append(f.mean(-1).mean(-1));
            x = x[:, :, :f.shape[2], :f.shape[3]] + f
        Ts = torch.cat(stoppers, dim=-1);
        leng = this.lenpred(Ts);
        if (t_override is None):
            t_override = leng.argmax(-1)[0];
        att = this.make_att(x, t_override);
        if (mask is not None):
            if (mask.shape[-2] != x.shape[-2] or mask.shape[-1] != x.shape[-1]):
                mask_ = trnf.interpolate(mask, [x.shape[-2], x.shape[-1]], mode="bilinear");
            else:
                mask_ = mask;
            att = att * mask_;
        return att, leng;
    def feat_aggr(this,input,mask,t_override):
        return this.att_len_core(input,mask,t_override);


    def forward(this, input_,mask=None,t_override=None):
        if this.detached:
            input=[this.semods[i](input_[i].detach()) for i in range(len(input_))]
        else:
            input=[this.semods[i](input_[i]) for i in range(len(input_))]
        return this.feat_aggr(input,mask,t_override);

    def feat_aggr_d(this, input, mask, t_override):
        return this.feat_aggr(input, mask, t_override);

    def forward_d(this, input_,mask=None,t_override=None):
        with torch.no_grad():
            input=[this.semods[i](input_[i]) for i in range(len(input_))]
            A,_=this.feat_aggr_d(input,mask,t_override);
        return A.detach().cpu();

class neko_CAM_stop_mp_seintr(neko_CAM_stop_seintr):

    def __init__(this, scales, maxT, depth, num_channels,num_se_channels,n_parts,detached=False):
        this.detached=detached;
        super(neko_CAM_stop_mp_seintr, this).__init__(scales, maxT*n_parts, depth, num_channels,num_se_channels);
        this.n_parts=n_parts;
    def feat_aggr(this,input,mask,t_override):
        x,leng=this.att_len_core(input,mask,t_override);
        x = x.reshape(x.shape[0], x.shape[1] // this.n_parts, this.n_parts, x.shape[2], x.shape[3]);
        return x,leng;


class neko_CAM_stop_mpf_seintr(neko_CAM_stop_seintr):

    def __init__(this, scales, maxT, depth, num_channels, num_se_channels, n_parts, detached=False):
        this.detached=detached;
        super(neko_CAM_stop_mpf_seintr, this).__init__(scales, maxT * (n_parts+1), depth, num_channels, num_se_channels);
        this.n_parts = n_parts;
    def feat_aggr(this,input,mask,t_override):
        x, leng=this.att_len_core(input,mask,t_override);
        x = x.reshape(x.shape[0], x.shape[1] // (this.n_parts+1), this.n_parts+1, x.shape[2], x.shape[3]);
        a=x[:,:,:1]*x[:,:,1:];
        return a,leng;
    def feat_aggr_d(this, input, mask, t_override):
        x, leng = this.att_len_core(input, mask, t_override);
        x = x.reshape(x.shape[0], x.shape[1] // (this.n_parts + 1), this.n_parts + 1, x.shape[2], x.shape[3]);
        return x,leng;

# The lite version.
# It assumes some "common location" of all parts.

class neko_CAM_stop_mpfl_seintr(neko_CAM_stop_seintr):

    def __init__(this, scales, maxT, depth, num_channels, num_se_channels, n_parts, detached=False):
        this.detached=detached;
        super(neko_CAM_stop_mpfl_seintr, this).__init__(scales, maxT +n_parts, depth, num_channels, num_se_channels);
        this.n_parts = n_parts;
    def feat_aggr(this,input,mask,t_override):
        x, leng=this.att_len_core(input,mask,t_override);
        c,p = x[:,this.n_parts:].unsqueeze(2),x[:,:this.n_parts].unsqueeze(1);
        a=c*p;
        return a,leng;
    def feat_aggr_d(this, input, mask, t_override):
        x, leng = this.att_len_core(input, mask, t_override);
        c, p = x[:, this.n_parts:].unsqueeze(2), x[:, :this.n_parts].unsqueeze(1);
        a = c * p;
        return a,leng;

class neko_CAM_stop_mpf_seintr_LN(neko_CAM_stop_seintr):
    def arm_last_deconv(this,deconvs,num_channels, nmasks,deconv_ksize,stride):
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, nmasks,
                                                        tuple(deconv_ksize),
                                                        tuple(stride),
                                                        (int(deconv_ksize[0] / 4.), int(deconv_ksize[1] / 4.))),
                                     nn.LayerNorm(num_channels),
                                     nn.Sigmoid()));


class neko_CAM_stop_mpf_seintr_IN(neko_CAM_stop_seintr):
    def arm_last_deconv(this,deconvs,num_channels, nmasks,deconv_ksize,stride):
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, nmasks,
                                                        tuple(deconv_ksize),
                                                        tuple(stride),
                                                        (int(deconv_ksize[0] / 4.), int(deconv_ksize[1] / 4.))),
                                     nn.InstanceNorm2d(num_channels),
                                     nn.Sigmoid()));
