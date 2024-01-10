from neko_2022_soai_zero.configs.modules.overlap_loss import  config_variance_dtcg3_loss
from neko_2022_soai_zero.aas_release.configs.loadouts.aas_mk2_common import arm_aasmk2_common
from neko_2022_soai_zero.aas_release.configs.loadouts.aas_mk2_ss import arm_mpfS_3sp_pm_viewpoint

def arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7g2_mpfS_3sp_pm_covl_dtcg3_32(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,wemb=0,wrej=0.1,inplace=True,wpart=1,drop=None):
    srcdst=arm_aasmk2_common(srcdst,prefix,capacity,feat_ch//4,tr_meta_path,expf,wemb,wrej,inplace,drop=drop);
    srcdst=arm_mpfS_3sp_pm_viewpoint(srcdst,prefix,capacity,maxT,feat_ch//4,expf=expf,detached=True,force_prototype_shape=[32,32]);
    # Well, this is kind abusing it, but never mind.
    srcdst[prefix+"part_loss"]=config_variance_dtcg3_loss(wpart);
    return srcdst;

def arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7g2_mpfS_3sp_pm_dtc32(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,wemb=0,wrej=0.1,inplace=True):
    srcdst=arm_aasmk2_common(srcdst,prefix,capacity,feat_ch//4,tr_meta_path,expf,wemb,wrej,inplace);
    srcdst=arm_mpfS_3sp_pm_viewpoint(srcdst,prefix,capacity,maxT,feat_ch//4,expf=expf,detached=True,force_prototype_shape=[32,32]);
    return srcdst;