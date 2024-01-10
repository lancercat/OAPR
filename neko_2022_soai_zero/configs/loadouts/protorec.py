
from neko_2022_soai_zero.configs.loadouts.base_model import arm_base_module_set,arm_base_module_setS;

from neko_2022_soai_zero.configs.modules.config_recon import config_dgrl
def arm_protorec(srcdst,prefix,feat_ch):
    srcdst[prefix+"p_recon"]=config_dgrl(feat_ch);
    return srcdst;

def arm_protorec_only_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,dropf=0,dropp=None):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,dropf=dropf,dropp=dropp);
    srcdst=arm_protorec(srcdst,prefix,feat_ch);
    return srcdst;


def arm_protorecS_only_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,dropf=0,dropp=None):
    srcdst=arm_base_module_setS(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,dropf=dropf,dropp=dropp);
    srcdst=arm_protorec(srcdst,prefix,feat_ch);
    return srcdst;

