# sampler_name, prototyper_name, feature_extractor_name, seq_name,
#                            CAMname, pred_name,ctx_name, loss_name,dom_loss_name,prec_loss_name,frec_loss_name,
#                            label_name, image_name, log_path, log_each, name, maxT
# srcdst["ctxmod"] = "NEP_skipped_NEP";
# srcdst["dom_mix"] = "NEP_skipped_NEP";
# srcdst["p_recon"] = "NEP_skipped_NEP";
# srcdst["f_recon"] = "NEP_skipped_NEP";
# return srcdst;
def arm_aas_routine(srcdst, prefix, routine_type,maxT, log_path, log_each,dsprefix,override_dict=None,rot_id="mjst",delay_till=None,xtra_preds=None):

    kvargs={
        "prototyper_name":prefix+"prototyper",
        "sampler_name":prefix+"Latin_62_sampler",
        "aam_name":prefix+"aam",
        "feature_extractor_name":prefix+"feature_extractor_cco",
        "CAMname":prefix+"TA",
        "seq_name":prefix+"DTD",
        "pred_name":[prefix+"pred"],
        "ctx_name": prefix + "ctxmod",
        "loss_name":[prefix+"loss_cls_emb"],
        "dom_loss_name": prefix+"dom_mix",
        "prec_loss_name": prefix+"p_recon",
        "frec_loss_name":  prefix+"f_recon",
        "part_loss_name": prefix+"part_loss",
        "image_name":dsprefix+"image",
        "label_name":dsprefix+"label",
        "log_path":log_path,
        "log_each":log_each,
        "name":prefix+rot_id,
        "maxT":maxT,
    }
    if(xtra_preds is not None):
        for i in range(len(xtra_preds)):
            kvargs["pred_name"].append(prefix+xtra_preds[i]);
            kvargs["loss_name"].append(prefix+"loss_cls_emb");
    if(delay_till is not None):
        kvargs["delay_till"]=delay_till;
    if(override_dict is not None):
        for k in override_dict:
            kvargs[k]=override_dict[k];
    srcdst[prefix+rot_id]= routine_type(
        **kvargs
    );

    srcdst[prefix+rot_id]["stream"]=prefix;
    return srcdst;

