
from neko_2022_soai_zero.configs.loadouts.base_model import arm_base_module_set
from neko_2022_soai_zero.configs.loadouts.individual_recs import arm_protorec_v2S, arm_v2_shufDPS, arm_v2_shufADPS, \
    arm_v2_frec
from neko_2022_soai_zero.configs.modules.config_recon import config_dcganN_insnorm


def arm_v2IN_frec_module_setDPS(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf=1, fecnt=3,
                                          wemb=0.3, wrej=0.1, inplace=True, detached_ta=False, detached_ga_proto=False,
                                          dropf=0, dropp=None, recon_speed=1, shuf_speed=1, rchunk=2, cchunk=2,
                                          frec_speed=1):
    srcdst = arm_base_module_set(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf, fecnt + 1, wemb, wrej,
                                 inplace, detached_ta=detached_ta, detached_ga_proto=detached_ga_proto, dropf=dropf,
                                 dropp=dropp);
    srcdst[prefix + "p_recon"] = config_dcganN_insnorm(feat_ch);
    # srcdst = arm_protorec_v2S(srcdst, prefix, feat_ch, speed=recon_speed);
    # srcdst = arm_v2_shufDPS(srcdst, prefix, feat_ch, speed=shuf_speed, rchunk=rchunk, cchunk=cchunk);
    srcdst = arm_v2_frec(srcdst,prefix,speed=frec_speed);
    return srcdst;


def arm_v2IN_prec_frec_module_setDPS(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf=1, fecnt=3,
                                          wemb=0.3, wrej=0.1, inplace=True, detached_ta=False, detached_ga_proto=False,
                                          dropf=0, dropp=None, recon_speed=1, shuf_speed=1, rchunk=2, cchunk=2,
                                          frec_speed=1):
    srcdst = arm_base_module_set(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf, fecnt + 1, wemb, wrej,
                                 inplace, detached_ta=detached_ta, detached_ga_proto=detached_ga_proto, dropf=dropf,
                                 dropp=dropp);
    srcdst[prefix + "p_recon"] = config_dcganN_insnorm(feat_ch);
    srcdst = arm_protorec_v2S(srcdst, prefix, feat_ch, speed=recon_speed);
    # srcdst = arm_v2_shufDPS(srcdst, prefix, feat_ch, speed=shuf_speed, rchunk=rchunk, cchunk=cchunk);
    srcdst = arm_v2_frec(srcdst,prefix,speed=frec_speed);
    return srcdst;

def arm_v2IN_prec_shuf_frec_module_setDPS(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf=1, fecnt=3,
                                          wemb=0.3, wrej=0.1, inplace=True, detached_ta=False, detached_ga_proto=False,
                                          dropf=0, dropp=None, recon_speed=1, shuf_speed=1, rchunk=2, cchunk=2,
                                          frec_speed=1):
    srcdst = arm_base_module_set(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf, fecnt + 1, wemb, wrej,
                                 inplace, detached_ta=detached_ta, detached_ga_proto=detached_ga_proto, dropf=dropf,
                                 dropp=dropp);
    srcdst[prefix + "p_recon"] = config_dcganN_insnorm(feat_ch);
    srcdst = arm_protorec_v2S(srcdst, prefix, feat_ch, speed=recon_speed);
    srcdst = arm_v2_shufDPS(srcdst, prefix, feat_ch, speed=shuf_speed, rchunk=rchunk, cchunk=cchunk);
    srcdst = arm_v2_frec(srcdst,prefix,speed=frec_speed);
    return srcdst;
def arm_v2IN_prec_shuf_frec_module_setADPS(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf=1, fecnt=3,
                                          wemb=0.3, wrej=0.1, inplace=True, detached_ta=False, detached_ga_proto=False,
                                          dropf=0, dropp=None, recon_speed=1, shuf_speed=1, rchunk=2, cchunk=2,
                                          frec_speed=1):
    srcdst = arm_base_module_set(srcdst, prefix, maxT, capacity, feat_ch, tr_meta_path, expf, fecnt + 1, wemb, wrej,
                                 inplace, detached_ta=detached_ta, detached_ga_proto=detached_ga_proto, dropf=dropf,
                                 dropp=dropp);
    srcdst[prefix + "p_recon"] = config_dcganN_insnorm(feat_ch);
    srcdst = arm_protorec_v2S(srcdst, prefix, feat_ch, speed=recon_speed);
    srcdst = arm_v2_shufADPS(srcdst, prefix, feat_ch, speed=shuf_speed, rchunk=rchunk, cchunk=cchunk);
    srcdst = arm_v2_frec(srcdst,prefix,speed=frec_speed);
    return srcdst;
