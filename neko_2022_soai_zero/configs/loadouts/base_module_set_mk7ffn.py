from  neko_2021_mjt.configs.modules.config_fe_db import config_fe_r45_binorm_orig;

def arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7ffn(srcdst,prefix,maxT,capacity,feat_ch,tr_meta_path,expf=1,fecnt=3,wemb=0.3,wrej=0.1,inplace=True):
    srcdst[prefix + "feature_extractor_container"] = config_fe_r45_binorm_orig(3, feat_ch,cnt=fecnt,expf=1,inplace=inplace);
    srcdst[prefix + "feature_extractor_cco"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res1")
    srcdst[prefix + "feature_extractor_proto"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res2")
    srcdst[prefix + "GA"] = config_sa_mk3(feat_ch=32);
    srcdst[prefix + "TA"] = config_cam_stop(maxT, feat_ch=feat_ch, scales=[
        [int(32), 16, 64],
        [int(128), 8, 32],
        [int(feat_ch), 8, 32]
    ]);
    srcdst = arm_rest_commonwop(srcdst, prefix, maxT, capacity, tr_meta_path,wemb=wemb,wrej=wrej)
    srcdst = arm_shared_prototyper_np(
        srcdst, prefix, capacity, feat_ch,
        prefix + "feature_extractor_proto",
        prefix + "GA",
        use_sp=False
    );
    return srcdst;