from neko_2021_mjt.configs.bogo_modules.config_back_sharing_prototyper import config_prototyper_gen3p, \
    config_prototyper_gen3pe, config_prototyper_gen3peT
from neko_2021_mjt.configs.modules.config_sp import config_sp_prototyper


def arm_shared_prototyper_np_part(srcdst,prefix,capacity, feat_ch,fe_name,cam_name,use_sp=True,force_proto_shape=None,nameoverride=None):
    if(use_sp):
        srcdst[prefix+"sp_proto"]=config_sp_prototyper(feat_ch,use_sp=use_sp);
    if(nameoverride is None):
        nameoverride=prefix+"prototyper";
    srcdst[nameoverride]=config_prototyper_gen3p(
        prefix+"sp_proto",
        fe_name,
        cam_name,
        None,
        capacity,
        force_proto_shape
    )
    return srcdst;
def arm_shared_prototyper_np_part_ex(srcdst,prefix,capacity, feat_ch,fe_name,cam_name,use_sp=True,force_proto_shape=None,nameoverride=None):
    if(use_sp):
        srcdst[prefix+"sp_proto"]=config_sp_prototyper(feat_ch,use_sp=use_sp);
    if(nameoverride is None):
        nameoverride=prefix+"prototyper";
    srcdst[nameoverride]=config_prototyper_gen3pe(
        prefix+"sp_proto",
        fe_name,
        cam_name,
        None,
        capacity,
        force_proto_shape
    )
    return srcdst;
def arm_shared_prototyper_np_part_exT(srcdst,prefix,capacity, feat_ch,fe_name,cam_name,use_sp=True,force_proto_shape=None,nameoverride=None):
    if(use_sp):
        srcdst[prefix+"sp_proto"]=config_sp_prototyper(feat_ch,use_sp=use_sp);
    if(nameoverride is None):
        nameoverride=prefix+"prototyper";
    srcdst[nameoverride]=config_prototyper_gen3peT(
        prefix+"sp_proto",
        fe_name,
        cam_name,
        None,
        capacity,
        force_proto_shape
    )
    return srcdst;
