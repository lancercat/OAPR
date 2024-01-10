from neko_2021_mjt.configs.bogo_modules.config_res_binorm import config_bogo_resbinorm;
from neko_2022_soai_zero.configs.loadouts.protorec_v2 import arm_base_module_set, config_dcganN_insnorm, \
    arm_protorec_v2S, arm_v2_shufDPS
from neko_2022_soai_zero.configs.modules.config_ch2fe import config_ch2fe
from neko_2022_soai_zero.configs.modules.config_inv import config_inv_loss_bi, config_inv_loss_uni


def arm_inv_loss_uni(srcdst,prefix,speed):
    srcdst[prefix+"inv_loss"]=config_inv_loss_uni(speed);
    return srcdst;


def arm_inv_loss_bi(srcdst, prefix,speed):
    srcdst[prefix + "inv_loss"] = config_inv_loss_bi(speed);
    return srcdst;


def arm_v2IN_prec_shuf_module_setDPS_uni(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,invspeed=1):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt+1,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
    srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
    srcdst[prefix+"recon_fe_bbn"]= config_bogo_resbinorm(prefix + "feature_extractor_container", "res4");
    srcdst[prefix+"recon_char_fe"]=config_ch2fe(prefix+"recon_fe_bbn",prefix+"GA",detached_ga=False);
    srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
    srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
    srcdst=arm_inv_loss_uni(srcdst,prefix,speed=invspeed);
    return srcdst;

def arm_v2IN_prec_shuf_module_setDPS_bi(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True,detached_ta=False,detached_ga_proto=False,dropf=0,dropp=None,recon_speed=1,shuf_speed=1,rchunk=2,cchunk=2,invspeed=1):
    srcdst=arm_base_module_set(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt+1,wemb,wrej,inplace,detached_ta=detached_ta,detached_ga_proto=detached_ga_proto,dropf=dropf,dropp=dropp);
    srcdst[prefix+"p_recon"]=config_dcganN_insnorm(feat_ch);
    srcdst[prefix+"recon_fe_bbn"]= config_bogo_resbinorm(prefix + "feature_extractor_container", "res4");
    srcdst[prefix+"recon_char_fe"]=config_ch2fe(prefix+"recon_fe_bbn",prefix+"GA",detached_ga=False);
    srcdst=arm_protorec_v2S(srcdst,prefix,feat_ch,speed=recon_speed);
    srcdst=arm_v2_shufDPS(srcdst,prefix,feat_ch,speed=shuf_speed,rchunk=rchunk,cchunk=cchunk);
    srcdst = arm_inv_loss_bi(srcdst, prefix,speed=invspeed);
    return srcdst;