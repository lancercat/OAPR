
from neko_2022_soai_zero.configs.loadouts.anc_def_p import arm_anc_defXL_affine_01_fe, arm_anc_def_affine_01_fe
from neko_2022_soai_zero.configs.loadouts.base_model import arm_common_part;
from neko_2022_soai_zero.configs.loadouts.domd import arm_dom_mix
from neko_2022_soai_zero.configs.loadouts.protorec import arm_protorec


def arm_full_model_affine_01(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True):
    srcdst=arm_anc_def_affine_01_fe(srcdst,prefix,feat_ch,inplace);
    srcdst=arm_common_part(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace);
    srcdst=arm_dom_mix(srcdst,prefix,feat_ch);
    srcdst=arm_protorec(srcdst,prefix,feat_ch);
    return srcdst;


def arm_fullXL_model_affine_01(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1.5,fecnt=3,wemb=0.3,wrej=0.1,inplace=True):
    srcdst=arm_anc_defXL_affine_01_fe(srcdst,prefix,feat_ch,inplace);
    TA_scales = [
        [int(64*expf), 16, 64],
        [int(256*expf), 8, 32],
        [int(feat_ch), 8, 32]];

    srcdst=arm_common_part(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace,TA_scales=TA_scales);
    srcdst=arm_dom_mix(srcdst,prefix,feat_ch);
    srcdst=arm_protorec(srcdst,prefix,feat_ch);
    return srcdst;

