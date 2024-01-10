# sampler_name, prototyper_name, feature_extractor_name, seq_name,
#                            CAMname, pred_name,ctx_name, loss_name,dom_loss_name,prec_loss_name,frec_loss_name,
#                            label_name, image_name, log_path, log_each, name, maxT
# srcdst["ctxmod"] = "NEP_skipped_NEP";
# srcdst["dom_mix"] = "NEP_skipped_NEP";
# srcdst["p_recon"] = "NEP_skipped_NEP";
# srcdst["f_recon"] = "NEP_skipped_NEP";
# return srcdst;
def arm_water_adapt_routine(srcdst, prefix, routine_type,maxT, log_path, log_each,dsprefix,override_dict=None,rot_id="mjst"):

    kvargs={
        "prototyper_name":prefix+"prototyper",
        "sampler_name":prefix+"Latin_62_sampler",
        "feature_extractor_name":prefix+"feature_extractor_cco",
        "CAMname":prefix+"TA",
        "seq_name":prefix+"DTD",
        "pred_name":[prefix+"pred"],
        "ctx_name": prefix + "ctxmod",
        "loss_name":[prefix+"loss_cls_emb"],
        "dom_loss_name": prefix+"dom_mix",
        "prec_loss_name": prefix+"p_recon",
        "frec_loss_name":  prefix+"f_recon",
        "image_name":dsprefix+"image",
        "label_name":dsprefix+"label",
        "beacon_name":dsprefix+"beacon",
        "bmask_name":dsprefix+"bmask",
        "log_path":log_path,
        "log_each":log_each,
        "name":prefix+rot_id,
        "maxT":maxT,}
    if(override_dict is not None):
        for k in override_dict:
            kvargs[k]=override_dict[k];
    srcdst[prefix+rot_id]= routine_type(
        **kvargs
    );

    srcdst[prefix+rot_id]["stream"]=prefix;
    return srcdst;
def arm_water2_adapt_routine(srcdst, prefix, routine_type,maxT, log_path, log_each,dsprefix,override_dict=None,rot_id="mjst"):
    kvargs = {
        "prototyper_name": prefix + "prototyper",
        "sampler_name": prefix + "Latin_62_sampler",
        "feature_extractor_name": prefix + "feature_extractor_cco",
        "CAMname": prefix + "TA",
        "seq_name": prefix + "DTD",
        "pred_name": [prefix + "pred"],
        "ctx_name": prefix + "ctxmod",
        "loss_name": [prefix + "loss_cls_emb"],
        "dom_loss_name": prefix + "dom_mix",
        "prec_name": prefix + "p_recon",
        "prec_loss_name": prefix + "p_recon_loss",
        "frec_name": prefix + "f_recon",
        "recon_char_fe_name": prefix + "recon_char_fe",
        "recon_char_pred_name": prefix + "recon_char_pred",
        "frec_loss_name": prefix + "f_recon_loss",
        "image_name": dsprefix + "image",
        "label_name": dsprefix + "label",
        "beacon_name": dsprefix + "beacon",
        "bmask_name": dsprefix + "bmask",
        "log_path": log_path,
        "log_each": log_each,
        "name": prefix + rot_id,
        "maxT": maxT, }
    if (override_dict is not None):
        for k in override_dict:
            kvargs[k] = override_dict[k];
    srcdst[prefix + rot_id] = routine_type(
        **kvargs
    );

    srcdst[prefix + rot_id]["stream"] = prefix;
    return srcdst;
