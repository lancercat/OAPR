import cv2
import numpy as np
import torch
from torch.nn import functional as trnf

from neko_sdk.ocr_modules.result_renderer import render_word


def dump_att_im(clip,GA,TA,length,gt,tdict=None,pr=None,scharset=None,sgmt=1):
    img = (clip.permute(1, 2, 0) * 255).detach().cpu().numpy();
    if(pr is not None):
        if(scharset is None):
            scharset=tdict;
        red, ned = render_word(tdict, scharset, img, gt.lower(), pr.lower());
    else:
        red=None;
    TA = trnf.interpolate(TA.unsqueeze(0), [32, 128]).squeeze(0).cpu();
    TIs = [clip.cpu()];
    if(GA is not None):
        GA = trnf.interpolate(GA.unsqueeze(0), [32, 128]).squeeze(0);
        GI = (GA.cpu() * 0.9 + 0.1) * clip.cpu();
        TIs.append(GI);

    for j in range(length):
        TIs.append(clip.cpu() * (TA[ j:j + 1] * 0.9 + 0.1));
    TIs.append(torch.max(TA[:length+1],dim=0)[0].repeat([3,1,1]));
    if(red is None):
        im=(torch.cat(TIs, 1).permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8);
    else:
        dim=(torch.cat(TIs, 1).permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8);
        dh=int(dim.shape[0]*(red.shape[1]/dim.shape[1]));
        im= np.concatenate([red,cv2.resize(dim,(red.shape[1],dh))]);
    for j in range(length//sgmt):
        im[32+j*32*sgmt:32+j*32*sgmt+3,:16,:]=(0,0,255);
    return im;

def dump_mask_im(clip,mask):
    ret = torch.cat([
        (clip.permute(1, 2, 0) * 255).detach(),
        (mask.repeat(3,1,1).permute(1, 2, 0) * 255).detach()],0);
    return ret.cpu().numpy();


def dump_att_im_grp(clip,GA,TAs,lengths,gt,tdict=None,pr=None,scharset=None,sgmts=None,stop=0):
    img = (clip.permute(1, 2, 0) * 255).detach().cpu().numpy();
    H,W=img.shape[0],img.shape[1];
    if(sgmts is None):
        sgmts=[1];
    TAlist=[trnf.interpolate(TA.unsqueeze(0), [H, W],mode="bilinear").squeeze(0).cpu() for TA in TAs]
    if(pr is not None):
        if(scharset is None):
            scharset=tdict;
        red, ned = render_word(tdict, scharset, img, gt.lower(), pr.lower());
    else:
        red=None;
    cnt = lengths[0] // sgmts[0];
    vatts=[clip.cpu()];
    if (GA is not None):
        GA = trnf.interpolate(GA.unsqueeze(0), [H, W]).squeeze(0);
        GI = (GA.cpu() * 0.9 + 0.1) * clip.cpu();
        vatts.append(GI);
    for i in range(cnt):
        first_in_timestamp=True;
        for agid in range(len(TAlist)):
            first_in_grp = True;
            TA=TAlist[agid];
            for j in range(sgmts[agid]):
                tim=(clip.cpu() * (TA[ i*sgmts[agid]+j:i*sgmts[agid] + j+1] * 0.9 + 0.1));
                if(first_in_grp):
                    tim[0, :3, :H] = 1;
                    tim[1, :3, :H] = 0;
                    tim[2, :3, :H] = 0;
                    first_in_grp=False;
                if(first_in_timestamp):
                    tim[0, :3, :H//2] = 0;
                    tim[1, :3, :H//2] = 0;
                    tim[2, :3, :H//2] = 1;
                    first_in_timestamp = False;
                vatts.append(tim)

    for agid in range(len(TAlist)):
        TA = TAlist[agid];
        vatts.append(torch.max(TA[:lengths[agid]+stop*sgmts[agid]], dim=0)[0].repeat([3, 1, 1]));
    if (red is None):
        im = (torch.clamp(torch.cat(vatts, 1).permute(1, 2, 0),0,1) * 255).detach().cpu().numpy().astype(np.uint8);
    else:
        dim = (torch.clamp(torch.cat(vatts, 1).permute(1, 2, 0),0,1)  * 255).detach().cpu().numpy().astype(np.uint8);
        dh = int(dim.shape[0] * (red.shape[1] / dim.shape[1]));
        im = np.concatenate([red, cv2.resize(dim, (red.shape[1], dh))]);
    return im;

def dump_att_ims(clips,GA,TAs,length_,gt,tdict=None,pr=None,scharset=None):
    tis=[];
    lengths=[];
    sgmts=[];
    TAlist=[];
    for TA_ in TAs:
        if(len(TA_.shape)==5):
            length=length_*TA_.shape[2];
            TA=TA_.reshape(TA_.shape[0],TA_.shape[1]*TA_.shape[2],TA_.shape[3],TA_.shape[4]);
            sgmt=TA_.shape[2];
        else:
            length=length_;
            sgmt=1;
            TA=TA_;
        lengths.append(length);
        sgmts.append(sgmt);
        TAlist.append(TA)

    for i in range(clips.shape[0]):
        if(GA is None):
            aga=None;
        else:
            aga=GA[i];
        if(gt is None):
            agt=None;
        else:
            agt=gt[i]
        tim=dump_att_im_grp(clips[i], aga, [TA[i] for TA in TAlist],
                        [length[i].item() for length in lengths],
                        agt,
                        sgmts=sgmts)
        tis.append(tim);
    return tis;
def dump_mask_ims(clips,maskes):
    tis=[];
    for i in range(clips.shape[0]):
        tis.append(dump_mask_im(clips[i],maskes[i]));
    return tis;

def debug_gta(clips,GA,TA,p_len,dbgkey):
    for i in range(clips.shape[0]):
        ti = dump_att_im(clips[i], GA[i], TA[i], p_len[i], None);
        cv2.namedWindow("a"+dbgkey, 0);
        cv2.imshow("a"+dbgkey, ti);
        cv2.waitKey(10);
# used in mk7 routines and later.
# this change allows va8 to provide a pixel level control over image splitting, hopefully can reduce some variance

def GTA2(modular_dict,masks_,length,features,GA,TA):
    if (TA.shape[-1] != GA.shape[-1]):
        sTA = trnf.interpolate(TA, [GA.shape[2], GA.shape[3]], mode="area");
    else:
        sTA = TA
    A = sTA * (GA * 0.9 + 0.1)
    out_emb = modular_dict["seq"](features[-1], A, length);
    return out_emb,A;

def temporal_attention_v1( clips,masks_, modular_dict, length,dbgname=None):
    features = modular_dict["feature_extractor"](clips)
    features = [f.contiguous() for f in features];

    TA, L = modular_dict["TA"](features);
    if (length is None):
        length= L.max(dim=-1)[1];
    GA = modular_dict["GA"](features);
    if (TA.shape[-1] != GA.shape[-1]):
        sTA=trnf.interpolate(TA,[GA.shape[2],GA.shape[3]],mode="area");
    else:
        sTA=TA
    A = sTA * GA;
    # dump_att(clips,GA,TA,length,None)
    out_emb = modular_dict["seq"](features[-1], A, length);
    if(dbgname is not None):
        if(length is not None):
            debug_gta(clips,GA,TA,length,dbgname);
        else:
            debug_gta(clips,GA,TA,L.max(dim=-1)[1],dbgname)
    return out_emb, A, L, GA, TA
def global_temporal_attention_v2( clips,masks_, modular_dict, length,dbgname=None):
    features = modular_dict["feature_extractor"](clips)
    features = [f.contiguous() for f in features];

    TA, L = modular_dict["TA"](features);
    if (length is None):
        length= L.max(dim=-1)[1];
    GA = modular_dict["GA"](features);
    if (TA.shape[-1] != GA.shape[-1]):
        sTA=trnf.interpolate(TA,[GA.shape[2],GA.shape[3]],mode="area");
    else:
        sTA=TA
    A = sTA * GA;
    # if we do not know the correct length, we use the predicted length.

    # dump_att(clips,GA,TA,length,None)
    out_emb = modular_dict["seq"](features[-1], A, length);

    if(dbgname is not None):
        debug_gta(clips,GA,TA,length,dbgname);
    return out_emb, A, L, GA, TA
def global_temporal_attention_v2dt( clips,masks_, modular_dict, length,dbgname=None):
    features = modular_dict["feature_extractor"](clips)
    features = [f.contiguous() for f in features];
    TA, L = modular_dict["TA"]([f.detach() for f in features]);
    GA = modular_dict["GA"](features);
    if (TA.shape[-1] != GA.shape[-1]):
        sTA = trnf.interpolate(TA, [GA.shape[2], GA.shape[3]], mode="area");
    else:
        sTA = TA
    A = sTA * GA;
    # dump_att(clips,GA,TA,length,None)
    out_emb = modular_dict["seq"](features[-1], A, length);
    if (dbgname is not None):
        debug_gta(clips, GA, TA, length,dbgname);
    return out_emb, A, L, GA, TA

def global_temporal_attention_v3dt( clips,masks_, modular_dict, length,dbgname=None):
    features = modular_dict["feature_extractor"](clips)
    features = [f.contiguous() for f in features];
    TA, L = modular_dict["TA"]([f.detach() for f in features]);
    if (length is None):
        length= L.max(dim=-1)[1];
    GA = modular_dict["GA"](features);
    if (TA.shape[-1] != GA.shape[-1]):
        sTA = trnf.interpolate(TA, [GA.shape[2], GA.shape[3]], mode="area");
    else:
        sTA = TA
    A = sTA * (GA*0.9+0.1)
    out_emb = modular_dict["seq"](features[-1], A, length);
    if (dbgname is not None):
        debug_gta(clips, GA, TA, length,dbgname);
    return out_emb, A, L, GA, TA

def global_temporal_attention_v3dt_va83( clips,masks_, modular_dict, length,dbgname=None):
    features = modular_dict["feature_extractor"](clips)
    features = [f.contiguous() for f in features];
    dfeatures=[f.detach() for f in features];
    GA = modular_dict["GA"](features);
    TA, L = modular_dict["TA"](dfeatures);
    if (length is None):
        length= L.max(dim=-1)[1];
    CTA, _ = modular_dict["CTA"](features);
    out_emb,A=GTA2(modular_dict,masks_,length, features, GA, TA);
    Cout_emb,CA=GTA2(modular_dict,masks_,length, features, GA, CTA);
    if (dbgname is not None):
        debug_gta(clips, GA, TA, length,dbgname);
    return out_emb, A, L, GA, TA,CTA,Cout_emb
def global_temporal_attention_v3dt_va83dcs( clips,masks_, modular_dict, length,dbgname=None):
    features = modular_dict["feature_extractor"](clips)
    features = [f.contiguous() for f in features];
    dfeatures=[f.detach() for f in features];
    GA = modular_dict["GA"](features);
    TA, L = modular_dict["TA"](dfeatures);
    if (length is None):
        length= L.max(dim=-1)[1];
    CTA, _ = modular_dict["CTA"](dfeatures,length);
    out_emb,A=GTA2(modular_dict,masks_,length, features, GA, TA);
    # let's keep GA.
    Cout_emb,CA=GTA2(modular_dict,masks_,length, dfeatures, GA, CTA);
    if (dbgname is not None):
        debug_gta(clips, GA, TA, length,dbgname);
    return out_emb, A, L, GA, TA,CTA,Cout_emb

def global_temporal_attention_v3dt_va83dc( clips,masks_, modular_dict, length,dbgname=None):
    features = modular_dict["feature_extractor"](clips)
    features = [f.contiguous() for f in features];
    dfeatures=[f.detach() for f in features];
    GA = modular_dict["GA"](features);
    TA, L = modular_dict["TA"](dfeatures);
    if (length is None):
        length= L.max(dim=-1)[1];
    CTA, _ = modular_dict["CTA"](dfeatures);
    out_emb,A=GTA2(modular_dict,masks_,length, features, GA, TA);
    Cout_emb,CA=GTA2(modular_dict,masks_,length, dfeatures, GA.detach(), CTA);
    if (dbgname is not None):
        debug_gta(clips, GA, TA, length,dbgname);
    return out_emb, A, L, GA, TA,CTA,Cout_emb

def global_temporal_attention_v4dt( clips,masks_, modular_dict, length,dbgname=None):
    features = modular_dict["feature_extractor"](clips)
    features = [f.contiguous() for f in features];
    TA, L = modular_dict["TA"]([f.detach() for f in features],length);
    if (length is None):
        length= L.max(dim=-1)[1];
    GA = modular_dict["GA"](features);
    if (TA.shape[-1] != GA.shape[-1]):
        sTA = trnf.interpolate(TA, [GA.shape[2], GA.shape[3]], mode="area");
    else:
        sTA = TA
    A = sTA * (GA*0.9+0.1)
    out_emb = modular_dict["seq"](features[-1], A, length);
    if (dbgname is not None):
        debug_gta(clips, GA, TA, length,dbgname);
    return out_emb, A, L, GA, TA

def global_temporal_attention_v3dtm( clips,masks_, modular_dict, length,dbgname=None):
    features = modular_dict["feature_extractor"](clips)
    features = [f.contiguous() for f in features];
    TA, L = modular_dict["TA"]([f.detach() for f in features]);
    if (length is None):
        length= L.max(dim=-1)[1];
    GA = modular_dict["GA"](features);
    if (TA.shape[-1] != GA.shape[-1]):
        sTA = trnf.interpolate(TA, [GA.shape[2], GA.shape[3]], mode="area");
    else:
        sTA = TA
    if(masks_.shape[-1]!=GA.shape[-1]):
        masks=trnf.interpolate(masks_, [GA.shape[2], GA.shape[3]], mode="area")
    else:
        masks=masks_
    A = sTA * (GA*0.9+0.1)*(masks);
    # dump_att(clips,GA,TA,length,None)
    out_emb = modular_dict["seq"](features[-1], A, length);
    if (dbgname is not None):
        debug_gta(clips, GA, TA, length,dbgname);
    return out_emb, A, L, GA, TA

def global_temporal_attention_v2dtfvb( clips, modular_dict, length,dbgname=None):
    modular_dict["feature_extractor"].model.freezebn();
    features = modular_dict["feature_extractor"](clips);
    modular_dict["feature_extractor"].model.unfreezebn();

    features = [f.contiguous() for f in features];
    TA, L = modular_dict["TA"]([f.detach() for f in features]);
    if (length is None):
        length= L.max(dim=-1)[1];
    GA = modular_dict["GA"](features);
    A = TA * GA;
    # dump_att(clips,GA,TA,length,None)
    out_emb = modular_dict["seq"](features[-1], A, length);
    if (dbgname is not None):
        debug_gta(clips, GA, TA, length,dbgname);
    return out_emb, A, L, GA, TA

