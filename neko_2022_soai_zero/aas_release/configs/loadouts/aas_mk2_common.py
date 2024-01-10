
from neko_2021_mjt.configs.common_subs.arm_post_fe_shared_prototyper import arm_shared_prototyper_np

from neko_2021_mjt.configs.modules.config_cam_stop import config_cam_stop
from neko_2021_mjt.configs.modules.config_cls_emb_loss import config_cls_emb_loss2
from neko_2021_mjt.configs.modules.config_dtd_xos_mk5 import config_dtdmk5
from neko_2021_mjt.configs.modules.config_ocr_sampler import config_ocr_sampler
from neko_2021_mjt.configs.modules.config_ospred import config_linxos
from neko_2021_mjt.configs.modules.config_sa import config_sa_mk3
from osocrNG.configs.typical_module_setups.feature_extraction.bogo_res45_family import config_bogo_resbinorm_g2_ffn
from osocrNG.configs.typical_module_setups.feature_extraction.config_r45_orig import config_res45_ffn_naive_core, \
    config_res45_core
from  neko_2021_mjt.configs.modules.config_fe_db import config_fe_r45_binorm_orig
from neko_2021_mjt.configs.bogo_modules.config_res_binorm import config_bogo_resbinorm

# As you see, this is designating configs to factories, there is no real modulars here.
def arm_aasmk2_common(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,wemb=0.3,wrej=0.1,inplace=True,drop=None):
    layer_factory_cfg,ffn_factory_cfg,bn_factory_cfg=config_res45_ffn_naive_core(3,feat_ch,expf,inplace,True,drop=drop);
    srcdst[prefix + "shared_conv_layers"]=layer_factory_cfg;
    srcdst[prefix + "sample_bn"]=bn_factory_cfg;
    srcdst[prefix + "proto_bn"]=bn_factory_cfg;
    srcdst[prefix + "shared_ffn_layers"]=ffn_factory_cfg;
    srcdst[prefix + "feature_extractor_cco"]=config_bogo_resbinorm_g2_ffn(
        prefix + "shared_conv_layers",prefix + "sample_bn",prefix+"shared_ffn_layers");
    srcdst[prefix + "feature_extractor_proto"]=config_bogo_resbinorm_g2_ffn(
        prefix + "shared_conv_layers",prefix + "proto_bn",prefix+"shared_ffn_layers")

    srcdst[prefix + "loss_cls_emb"] = config_cls_emb_loss2(wemb, wrej);
    srcdst[prefix + "Latin_62_sampler"] = config_ocr_sampler(tr_meta_path, capacity);

    srcdst[prefix +"ctxmod"]="NEP_skipped_NEP";
    srcdst[prefix +"dom_mix"]="NEP_skipped_NEP";
    srcdst[prefix +"p_recon"]="NEP_skipped_NEP";
    srcdst[prefix +"f_recon"]="NEP_skipped_NEP";
    srcdst[prefix + "aam"]="NEP_skipped_NEP";
    srcdst[prefix +"part_loss"]="NEP_skipped_NEP";
    return srcdst;
def arm_aasmk2_commonO(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,wemb=0.0,wrej=0.1,inplace=True,dirty_frac=0.1,too_simple_frac=0.1):
    layer_factory_cfg,ffn_factory_cfg,bn_factory_cfg=config_res45_ffn_naive_core(3,feat_ch,expf,inplace,True);
    srcdst[prefix + "shared_conv_layers"]=layer_factory_cfg;
    srcdst[prefix + "sample_bn"]=bn_factory_cfg;
    srcdst[prefix + "proto_bn"]=bn_factory_cfg;
    srcdst[prefix + "shared_ffn_layers"]=ffn_factory_cfg;
    srcdst[prefix + "feature_extractor_cco"]=config_bogo_resbinorm_g2_ffn(
        prefix + "shared_conv_layers",prefix + "sample_bn",prefix+"shared_ffn_layers");
    srcdst[prefix + "feature_extractor_proto"]=config_bogo_resbinorm_g2_ffn(
        prefix + "shared_conv_layers",prefix + "proto_bn",prefix+"shared_ffn_layers")

    srcdst[prefix + "loss_cls_emb"] = config_cls_emb_lossohem(too_simple_frac=too_simple_frac,dirty_frac=dirty_frac);
    srcdst[prefix + "Latin_62_sampler"] = config_ocr_sampler(tr_meta_path, capacity);

    srcdst[prefix +"ctxmod"]="NEP_skipped_NEP";
    srcdst[prefix +"dom_mix"]="NEP_skipped_NEP";
    srcdst[prefix +"p_recon"]="NEP_skipped_NEP";
    srcdst[prefix +"f_recon"]="NEP_skipped_NEP";
    srcdst[prefix + "aam"]="NEP_skipped_NEP";
    srcdst[prefix +"part_loss"]="NEP_skipped_NEP";
    return srcdst;

def arm_aasmk2noffn_common(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,wemb=0.3,wrej=0.1,inplace=True):
    layer_factory_cfg,bn_factory_cfg=config_res45_core(3,feat_ch,expf,inplace,True);
    srcdst[prefix + "shared_conv_layers"]=layer_factory_cfg;
    srcdst[prefix + "sample_bn"]=bn_factory_cfg;
    srcdst[prefix + "proto_bn"]=bn_factory_cfg;
    srcdst[prefix + "feature_extractor_cco"]=config_bogo_resbinorm_g2(
        prefix + "shared_conv_layers",prefix + "sample_bn");
    srcdst[prefix + "feature_extractor_proto"]=config_bogo_resbinorm_g2(
        prefix + "shared_conv_layers",prefix + "proto_bn")

    srcdst[prefix + "loss_cls_emb"] = config_cls_emb_loss2(wemb, wrej);
    srcdst[prefix + "Latin_62_sampler"] = config_ocr_sampler(tr_meta_path, capacity);

    srcdst[prefix +"ctxmod"]="NEP_skipped_NEP";
    srcdst[prefix +"dom_mix"]="NEP_skipped_NEP";
    srcdst[prefix +"p_recon"]="NEP_skipped_NEP";
    srcdst[prefix +"f_recon"]="NEP_skipped_NEP";
    srcdst[prefix + "aam"]="NEP_skipped_NEP";
    srcdst[prefix +"part_loss"]="NEP_skipped_NEP";
    return srcdst;

def arm_aasmk2_common_old(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,wemb=0.3,wrej=0.1,inplace=True):
    srcdst[prefix + "feature_extractor_container"] = config_fe_r45_binorm_orig(3, feat_ch, cnt=2, expf=1,
                                                                               inplace=inplace);
    srcdst[prefix + "feature_extractor_cco"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res1")
    srcdst[prefix + "feature_extractor_proto"] = config_bogo_resbinorm(prefix + "feature_extractor_container", "res2")

    srcdst[prefix + "loss_cls_emb"] = config_cls_emb_loss2(wemb, wrej);
    srcdst[prefix + "Latin_62_sampler"] = config_ocr_sampler(tr_meta_path, capacity);

    srcdst[prefix + "ctxmod"] = "NEP_skipped_NEP";
    srcdst[prefix + "dom_mix"] = "NEP_skipped_NEP";
    srcdst[prefix + "p_recon"] = "NEP_skipped_NEP";
    srcdst[prefix + "f_recon"] = "NEP_skipped_NEP";
    srcdst[prefix + "part_loss"] = "NEP_skipped_NEP";
    srcdst[prefix + "aam"] = "NEP_skipped_NEP";
    return srcdst;


def arm_aa_dagrn(srcdst, prefix,hardness=3,has_color_jitter=True,has_spatial_jitter=True):
    srcdst[prefix + "aam"]=config_dagrn(3,has_color_jitter, has_spatial_jitter, (0.125, 0.0625),hardness=hardness);
    return srcdst;

def arm_aa_dagrnIN(srcdst, prefix,hardness=3,has_color_jitter=True,has_spatial_jitter=True):
    srcdst[prefix + "aam"]=config_dagrnIN(3,has_color_jitter, has_spatial_jitter, (0.125, 0.0625),hardness=hardness);
    return srcdst;
def arm_aa_dagrnINv2(srcdst, prefix,location_hardness=10,color_hardness=0.1):
    srcdst[prefix + "aam"]=config_dagrnINv2(3, (0.125, 0.0625),location_hardness=location_hardness,color_hardness=color_hardness);
    return srcdst;

def arm_aa_dagrnINv3(srcdst, prefix,location_hardness=10,color_hardness=0.1):
    srcdst[prefix + "aam"]=config_dagrnINv3(3, (0.125, 0.0625),location_hardness=location_hardness,color_hardness=color_hardness);
    return srcdst;
def arm_aa_dagrnINv4(srcdst, prefix,location_hardness=10,color_hardness=0.1):
    srcdst[prefix + "aam"]=config_dagrnINv3(3, (0.25, 0.25),location_hardness=location_hardness,color_hardness=color_hardness);
    return srcdst;
def arm_aa_dagrnINv5(srcdst, prefix,location_hardness=10,color_hardness=0.1):
    srcdst[prefix + "aam"]=config_dagrnINv4(3, (0.25, 0.25),location_hardness=location_hardness,color_hardness=color_hardness);
    return srcdst;
def arm_baseline_viewpoint(srcdst,prefix,capacity,maxT,feat_ch):
    srcdst[prefix + "TA"] = config_cam_stop(maxT, feat_ch=feat_ch, scales=[
        [int(32), 16, 64],
        [int(128), 8, 32],
        [int(feat_ch), 8, 32],
    ]);
    srcdst[prefix + "DTD"] = config_dtdmk5();
    srcdst = arm_shared_prototyper_np(
        srcdst, prefix, capacity, feat_ch,
        prefix + "feature_extractor_proto",
        prefix + "GA",
        use_sp=False
    );
    srcdst[prefix + "GA"] = config_sa_mk3(feat_ch=32);
    srcdst[prefix + "pred"] = config_linxos();
    return srcdst;
from neko_2022_soai_zero.configs.modules.config_cam_stop_se import config_cam_stop_seintr
from neko_2022_soai_zero.configs.modules.config_sa_se import config_sa_mk3_seintr
def arm_baseline_viewpointS(srcdst,prefix,capacity,maxT,feat_ch,detached=False,force_prototype_shape=None):
    srcdst[prefix + "TA"] = config_cam_stop_seintr(maxT, feat_ch=feat_ch, scales=[
        [int(32)+32, 16, 64],
        [int(128)+32, 8, 32],
        [int(feat_ch)+32, 8, 32],
    ],detached=detached);
    srcdst[prefix + "DTD"] = config_dtdmk5();
    srcdst = arm_shared_prototyper_np(
        srcdst, prefix, capacity, feat_ch,
        prefix + "feature_extractor_proto",
        prefix + "GA",
        use_sp=False,
        force_proto_shape=force_prototype_shape
    );
    srcdst[prefix + "GA"] = config_sa_mk3_seintr(feat_ch=32,detached=True);
    srcdst[prefix + "pred"] = config_linxos();
    return srcdst;

