
from neko_2022_soai_zero.configs.loadouts.base_model import arm_base_module_set;
from neko_2022_soai_zero.configs.loadouts.domd import arm_dom_mix, arm_dom_mix_slacky, arm_dom_mix_slacky_mva, \
    arm_dom_mix_mva
from neko_2022_soai_zero.configs.loadouts.protorec import arm_protorec;


def arm_protorec_dommix_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace);
    srcdst=arm_protorec(srcdst,prefix,feat_ch);
    srcdst=arm_dom_mix(srcdst,prefix,feat_ch);
    return srcdst;

def arm_protorec_dommix_slacky_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,speed=1.0):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace);
    srcdst=arm_protorec(srcdst,prefix,feat_ch);
    srcdst=arm_dom_mix_slacky(srcdst,prefix,feat_ch,speed);
    return srcdst;

def arm_protorec_mva_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace);
    srcdst=arm_protorec(srcdst,prefix,feat_ch);
    srcdst=arm_dom_mix_mva(srcdst,prefix);
    return srcdst;


def arm_protorec_dommix_slacky_mva_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detach_TA=False,detach_GA_proto=False,speed=0.5):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,detached_ta=detach_TA,detached_ga_proto=detach_GA_proto);
    srcdst=arm_protorec(srcdst,prefix,feat_ch);
    srcdst=arm_dom_mix_slacky_mva(srcdst,prefix,feat_ch,speed=speed);
    return srcdst;

