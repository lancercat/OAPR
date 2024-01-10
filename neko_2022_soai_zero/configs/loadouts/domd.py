
from neko_2022_soai_zero.configs.loadouts.base_model import arm_base_module_set;

from neko_2022_soai_zero.configs.modules.config_dommix import config_dmce,config_dmmd,config_dmceS,config_dmmva,config_dmceS_mva
def arm_dom_mix(srcdst,prefix,feat_ch):
    srcdst[prefix+"dom_mix"]=config_dmce(feat_ch,2);
    return srcdst;
def arm_dom_mix_slacky(srcdst,prefix,feat_ch,speed=1.0,domcnt=2):
    srcdst[prefix+"dom_mix"]=config_dmceS(feat_ch,domcnt,speed);
    return srcdst;
def arm_dom_mix_slacky_mva(srcdst,prefix,feat_ch,speed=1):
    srcdst[prefix+"dom_mix"]=config_dmceS_mva(feat_ch,2,speed=speed);
    return srcdst;

def arm_dom_mix_mva(srcdst,prefix,speed=1,detach_policy="None"):
    srcdst[prefix+"dom_mix"]=config_dmmva(speed=speed,detach_policy=detach_policy);
    return srcdst;
def arm_dom_mix_dist_of_mean(srcdst,prefix,feat_ch):
    srcdst[prefix+"dom_mix"]=config_dmmd(feat_ch,2);
    return srcdst;
def arm_dom_mix_only_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace);
    srcdst=arm_dom_mix(srcdst,prefix,feat_ch);
    return srcdst;
def arm_dom_mix_mva_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace);
    srcdst=arm_dom_mix_mva(srcdst,prefix);
    return srcdst;
def arm_dom_mix_slacky_only_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,speed=1.0):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace);
    srcdst=arm_dom_mix_slacky(srcdst,prefix,feat_ch,speed);
    return srcdst;


def arm_dom_mix_DoM_only_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace);
    srcdst=arm_dom_mix_dist_of_mean(srcdst,prefix,feat_ch);
    return srcdst;

