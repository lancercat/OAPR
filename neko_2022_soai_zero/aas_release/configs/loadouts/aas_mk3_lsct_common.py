
from neko_2021_mjt.configs.modules.config_cls_emb_loss import config_cls_emb_loss2
from neko_2021_mjt.configs.modules.config_ocr_sampler import config_ocr_sampler
from neko_2022_soai_zero.configs.modules.config_cam_stop_se import config_cam_stop_mpf_seintr
from neko_2022_soai_zero.configs.modules.config_sa_se import config_sa_mk3_seintr
from neko_2022_soai_zero.configs.modules.config_softlink import config_softlink
from osocrNG.configs.typical_module_setups.feature_extraction.bogo_res45_family import config_bogo_resbinorm_g2_ffn


def arm_forked_lsct_ST_SG_etc(srcdst,prefix,capacity,maxT,feat_ch,expf,detached=False,force_proto_shape=False):


    srcdst[prefix + "lsct_"+"TA"] = config_softlink(
        prefix + "TA"
    );
    srcdst[prefix+"lsct_"+"DTD"]=config_softlink(
        prefix + "DTD"
    );
    srcdst[prefix+"lsct_"+"GA"]=config_softlink(
        prefix+"GA"
    );
    srcdst[prefix + "lsct_" + "pred"] = config_softlink(
        prefix + "pred"
    );
    srcdst[prefix+ "lsct_" +"ctxmod"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"dom_mix"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"p_recon"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"f_recon"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"aam"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"part_loss"]="NEP_skipped_NEP";
    return srcdst;

def arm_forked_lsct_DT_SG_etc(srcdst,prefix,capacity,maxT,feat_ch,expf,detached=False,force_proto_shape=False):

    srcdst[prefix + "lsct_"+"TA"] =  config_cam_stop_mpf_seintr(maxT, feat_ch=feat_ch, scales=[
        [int(32*expf)+32, 16, 64],
        [int(128*expf)+32, 8, 32],
        [int(feat_ch)+32, 8, 32],
    ], n_parts=4,detached=detached);
    srcdst[prefix+"lsct_"+"DTD"]=config_softlink(
        prefix + "DTD"
    );
    srcdst[prefix+"lsct_"+"GA"]=config_softlink(
        prefix+"GA"
    );
    srcdst[prefix + "lsct_" + "pred"] = config_softlink(
        prefix + "pred"
    );
    srcdst[prefix+ "lsct_" +"ctxmod"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"dom_mix"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"p_recon"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"f_recon"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"aam"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"part_loss"]="NEP_skipped_NEP";
    return srcdst;


def arm_forked_lsct_ST_DG_etc(srcdst,prefix,capacity,maxT,feat_ch,expf,detached=False,force_proto_shape=False):

    srcdst[prefix + "lsct_"+"TA"] = config_softlink(
        prefix + "TA"
    );
    srcdst[prefix+"lsct_"+"DTD"]=config_softlink(
        prefix + "DTD"
    );
    srcdst[prefix+"lsct_"+"GA"]=config_sa_mk3_seintr(feat_ch=int(expf*32), n_parts=4,detached=detached);
    srcdst[prefix + "lsct_" + "pred"] = config_softlink(
        prefix + "pred"
    );
    srcdst[prefix+ "lsct_" +"ctxmod"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"dom_mix"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"p_recon"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"f_recon"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"aam"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"part_loss"]="NEP_skipped_NEP";
    return srcdst;


def arm_forked_lsct_DT_DG_etc(srcdst,prefix,capacity,maxT,feat_ch,expf,detached=False,force_proto_shape=False):

    srcdst[prefix + "lsct_"+"TA"] =  config_cam_stop_mpf_seintr(maxT, feat_ch=feat_ch, scales=[
        [int(32*expf)+32, 16, 64],
        [int(128*expf)+32, 8, 32],
        [int(feat_ch)+32, 8, 32],
    ], n_parts=4,detached=detached);
    srcdst[prefix+"lsct_"+"DTD"]=config_softlink(
        prefix + "DTD"
    );
    srcdst[prefix + "lsct_" + "GA"] = config_sa_mk3_seintr(feat_ch=int(32*expf), n_parts=4, detached=detached);
    srcdst[prefix + "lsct_" + "pred"] = config_softlink(
        prefix + "pred"
    );
    srcdst[prefix+ "lsct_" +"ctxmod"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"dom_mix"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"p_recon"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"f_recon"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"aam"]="NEP_skipped_NEP";
    srcdst[prefix + "lsct_"+"part_loss"]="NEP_skipped_NEP";
    return srcdst;


def arm_aasmk3_lsct_dbn(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,wemb=0.3,wrej=0.1,inplace=True):
    _,_,bn_factory_cfg=config_res45_ffn_naive_core(3,feat_ch,expf,inplace,True);
    srcdst[prefix +"lsct_"+ "proto_bn"]=bn_factory_cfg;
    srcdst[prefix +"lsct_sample_bn"+ "proto_bn"]=bn_factory_cfg;
    srcdst[prefix +"lsct_"+ "feature_extractor_cco"]=config_bogo_resbinorm_g2_ffn(
        prefix + "shared_conv_layers",prefix + "lsct_sample_bn",prefix+"shared_ffn_layers");
    srcdst[prefix +"lsct_"+ "feature_extractor_proto"]=config_bogo_resbinorm_g2_ffn(
        prefix + "shared_conv_layers",prefix + "lsct_proto_bn",prefix+"shared_ffn_layers")

def arm_aasmk3_lsct_sbn(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,wemb=0.3,wrej=0.1,inplace=True):
    srcdst[prefix +"lsct_"+ "feature_extractor_cco"]=config_softlink(prefix+ "feature_extractor_cco");
    srcdst[prefix +"lsct_"+ "feature_extractor_proto"]=config_softlink(prefix+ "feature_extractor_proto");

    srcdst[prefix +"lsct_"+ "loss_cls_emb"] = config_cls_emb_loss2(wemb, wrej);
    return srcdst;

def arm_aasmk3_lsct_dp(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,wemb=0.3,wrej=0.1,inplace=True):
    srcdst[prefix +"lsct_"+ "feature_extractor_cco"]=config_softlink(prefix+ "feature_extractor_cco");
    _,_,bn_factory_cfg=config_res45_ffn_naive_core(3,feat_ch,expf,inplace,True);
    srcdst[prefix +"lsct_"+ "proto_bn"]=bn_factory_cfg;
    srcdst[prefix + "lsct_" + "feature_extractor_proto"] = config_bogo_resbinorm_g2_ffn(
        prefix + "shared_conv_layers", prefix + "lsct_proto_bn", prefix + "shared_ffn_layers");
    srcdst[prefix +"lsct_"+ "loss_cls_emb"] = config_cls_emb_loss2(wemb, wrej);

    return srcdst;

def arm_aasmk3_lsct_df(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,wemb=0.3,wrej=0.1,inplace=True):
    _,_,bn_factory_cfg=config_res45_ffn_naive_core(3,feat_ch,expf,inplace,True);
    srcdst[prefix + "lsct_" + "sample_bn"] = bn_factory_cfg;
    srcdst[prefix + "lsct_" + "feature_extractor_cco"] = config_bogo_resbinorm_g2_ffn(
        prefix + "shared_conv_layers", prefix + "lsct_sample_bn", prefix + "shared_ffn_layers");
    srcdst[prefix +"lsct_"+ "loss_cls_emb"] = config_cls_emb_loss2(wemb, wrej);
    return srcdst;

def arm_aasmk3_lsct_dp_df(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,wemb=0.3,wrej=0.1,inplace=True):
    _,_,bn_factory_cfg=config_res45_ffn_naive_core(3,feat_ch,expf,inplace,True);

    srcdst[prefix + "lsct_" + "proto_bn"] = bn_factory_cfg;
    srcdst[prefix + "lsct_" + "feature_extractor_proto"] = config_bogo_resbinorm_g2_ffn(
        prefix + "shared_conv_layers", prefix + "lsct_proto_bn", prefix + "shared_ffn_layers")

    srcdst[prefix + "lsct_" + "sample_bn"] = bn_factory_cfg;
    srcdst[prefix + "lsct_" + "feature_extractor_cco"] = config_bogo_resbinorm_g2_ffn(
        prefix + "shared_conv_layers", prefix + "lsct_sample_bn", prefix + "shared_ffn_layers");



    srcdst[prefix +"lsct_"+ "loss_cls_emb"] = config_cls_emb_loss2(wemb, wrej);

    return srcdst;

def arm_aasmk3_lsct_common(srcdst,prefix,capacity,feat_ch,tr_meta_path,expf=1,wemb=0.3,wrej=0.1,inplace=True):
    layer_factory_cfg,ffn_factory_cfg,bn_factory_cfg=config_res45_ffn_naive_core(3,feat_ch,expf,inplace,True);
    srcdst[prefix + "shared_conv_layers"]=layer_factory_cfg;
    srcdst[prefix + "sample_bn"]=bn_factory_cfg;
    srcdst[prefix + "proto_bn"]=bn_factory_cfg;
    srcdst[prefix + "shared_ffn_layers"]=ffn_factory_cfg;
    srcdst[prefix + "feature_extractor_cco"]=config_bogo_resbinorm_g2_ffn(
        prefix + "shared_conv_layers",prefix + "sample_bn",prefix+"shared_ffn_layers");
    srcdst[prefix + "feature_extractor_proto"]=config_bogo_resbinorm_g2_ffn(
        prefix + "shared_conv_layers",prefix + "proto_bn",prefix+"shared_ffn_layers");

    srcdst[prefix + "loss_cls_emb"] = config_cls_emb_loss2(wemb, wrej);
    srcdst[prefix + "Latin_62_sampler"] = config_ocr_sampler(tr_meta_path, capacity);

    srcdst[prefix +"ctxmod"]="NEP_skipped_NEP";
    srcdst[prefix +"dom_mix"]="NEP_skipped_NEP";
    srcdst[prefix +"p_recon"]="NEP_skipped_NEP";
    srcdst[prefix +"f_recon"]="NEP_skipped_NEP";
    srcdst[prefix +"aam"]="NEP_skipped_NEP";
    srcdst[prefix +"part_loss"]="NEP_skipped_NEP";
    return srcdst;
