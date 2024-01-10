from neko_2021_mjt.configs.common_subs.arm_post_fe_shared_prototyper import arm_shared_prototyper_np
from neko_2022_soai_zero.aas_release.configs.loadouts.aas_mk3_lsct_common import arm_forked_lsct_ST_SG_etc, \
    arm_aasmk3_lsct_sbn;
from neko_2022_soai_zero.aas_release.configs.loadouts.aas_mk2_common import arm_aasmk2_common, \
    arm_baseline_viewpointS


def arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7g2S_dtc32(srcdst, prefix, maxT, capacity, feat_ch,
                                                                          tr_meta_path, expf=1, wemb=0, wrej=0.1,
                                                                          inplace=True):
    srcdst = arm_aasmk2_common(srcdst, prefix, capacity, feat_ch, tr_meta_path, expf, wemb, wrej, inplace);
    srcdst = arm_baseline_viewpointS(srcdst, prefix, capacity, maxT, feat_ch, detached=True,
                                     force_prototype_shape=[32, 32]);
    return srcdst

def arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7g2S_dtc32_AP(srcdst, prefix, maxT, capacity, feat_ch,
                                                                          tr_meta_path, expf=1, wemb=0, wrej=0.1,
                                                                          inplace=True):
    srcdst = arm_aasmk2_common(srcdst, prefix, capacity, feat_ch, tr_meta_path, expf, wemb, wrej, inplace);
    srcdst = arm_baseline_viewpointS(srcdst, prefix, capacity, maxT, feat_ch, detached=True,
                                     force_prototype_shape=[32, 32]);
    return srcdst

def arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7g2S_dtc48(srcdst, prefix, maxT, capacity, feat_ch,
                                                                          tr_meta_path, expf=1, wemb=0, wrej=0.1,
                                                                          inplace=True):
    srcdst = arm_aasmk2_common(srcdst, prefix, capacity, feat_ch, tr_meta_path, expf, wemb, wrej, inplace);
    srcdst = arm_baseline_viewpointS(srcdst, prefix, capacity, maxT, feat_ch, detached=True,
                                     force_prototype_shape=[48, 48]);
    return srcdst


def arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7g2S_dtc32_lsct_share_all(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,wemb=0,wrej=0.1,inplace=True):
    srcdst=arm_aasmk2_common(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf,wemb,wrej,inplace);
    srcdst=arm_baseline_viewpointS(srcdst,prefix,capacity,maxT,feat_ch,detached=True,force_prototype_shape=[32,32]);
    srcdst=arm_aasmk3_lsct_sbn(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf,wemb,wrej,inplace);
    srcdst=arm_forked_lsct_ST_SG_etc(srcdst,prefix,capacity,maxT,feat_ch,expf)
    srcdst = arm_shared_prototyper_np(
        srcdst, prefix + "lsct_", capacity, feat_ch,
                prefix + "lsct_" + "feature_extractor_proto",
                prefix + "lsct_" + "GA",
        use_sp=False
    );
    return srcdst;