
from neko_2021_mjt.configs.bogo_modules.config_res_binorm import config_bogo_resbinorm;
from neko_2022_soai_zero.configs.loadouts.base_model import arm_common_part;
from neko_2022_soai_zero.configs.modules.controlled_def_fe_affine_01 import config_fe_r45_binorm_orig_cdef_affine_01


def arm_anc_def_affine_01_fe(srcdst,prefix,feat_ch,inplace=True):
    # srcdst["assess_res1"]=config_neko_tinyfpn_assess();
    # srcdst["ctrl_def_res1"]=config_neko_
    # srcdst["ctrl_res2"] = config_nulldefault_def();
    srcdst[prefix + "feature_extractor_container"] = config_fe_r45_binorm_orig_cdef_affine_01(3, feat_ch,cnt=2,expf=1,inplace=inplace);
    srcdst[prefix + "feature_extractor_cco"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res1");
    srcdst[prefix + "feature_extractor_proto"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res2");
    return srcdst;
def arm_anc_def_only_module_set_affine_01(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True):
    srcdst=arm_anc_def_affine_01_fe(srcdst,prefix,feat_ch,inplace);
    srcdst=arm_common_part(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace);
    return srcdst;



def arm_anc_defXL_affine_01_fe(srcdst,prefix,feat_ch,inplace=True,expf=1.5):
    # srcdst["assess_res1"]=config_neko_tinyfpn_assess();
    # srcdst["ctrl_def_res1"]=config_neko_
    # srcdst["ctrl_res2"] = config_nulldefault_def();
    ochs = [int(32 * expf), int(64 * expf), int(128 * expf), int(256 * expf), int(512 * expf), feat_ch];
    srcdst[prefix + "feature_extractor_container"] = config_fe_r45_binorm_orig_cdef_affine_01(3, feat_ch,cnt=2,expf=1,inplace=inplace,ochs=ochs);
    srcdst[prefix + "feature_extractor_cco"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res1");
    srcdst[prefix + "feature_extractor_proto"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res2");
    return srcdst;
def arm_anc_def_onlyXL_module_set_affine_01(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True):
    srcdst=arm_anc_defXL_affine_01_fe(srcdst,prefix,feat_ch,inplace);
    srcdst=arm_common_part(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf,fecnt,wemb,wrej,inplace);
    return srcdst;


