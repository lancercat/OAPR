# segment similarity
from neko_2021_mjt.configs.modules.config_dtd_xos_mk5 import config_dtdmk5ce
from neko_2021_mjt.configs.modules.config_ospred import config_linxosPM

from neko_2022_soai_zero.configs.modules.config_cam_stop_se import config_cam_stop_mpf_seintr
from neko_2022_soai_zero.configs.modules.config_sa_rect import config_sa_rect
from neko_2022_soai_zero.configs.modules.config_sa_se import config_sa_mk3_seintr
from neko_2022_soai_zero.configs.modules.overlap_loss import  config_variance_dtcg3_loss
from neko_2022_soai_zero.aas_release.configs.common_subs.shared_prototyper_np_part import \
    arm_shared_prototyper_np_part_ex
from neko_2022_soai_zero.aas_release.configs.loadouts.aas_mk2_common import \
    arm_aasmk2_common, arm_aasmk2_common_old, arm_aasmk2_commonO, arm_aasmk2noffn_common

def arm_mpfS_3sp_pm_viewpoint(srcdst,prefix,capacity,maxT,feat_ch,expf,detached=False,force_prototype_shape=None):
    srcdst[prefix + "TA"] = config_cam_stop_mpf_seintr(maxT, feat_ch=feat_ch, scales=[
        [int(32*expf)+32, 16, 64],
        [int(128*expf)+32, 8, 32],
        [int(feat_ch)+32, 8, 32],
    ], n_parts=4,detached=detached);

    srcdst[prefix + "GA"] = config_sa_mk3_seintr(feat_ch=int(expf*32), n_parts=4,detached=detached);
    srcdst[prefix + "pred"] = config_linxosPM();
    srcdst[prefix + "DTD"] = config_dtdmk5ce();
    srcdst = arm_shared_prototyper_np_part_ex(
        srcdst, prefix, capacity, feat_ch,
        prefix + "feature_extractor_proto",
        prefix + "GA",
        use_sp=False,
        force_proto_shape=force_prototype_shape
    );
    return srcdst;
