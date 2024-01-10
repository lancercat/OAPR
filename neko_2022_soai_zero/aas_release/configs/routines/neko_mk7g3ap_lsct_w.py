from neko_2022_soai_zero.aas_release.routines.neko_mk7g3ap_lsct_word import neko_HDOS2C_routine_CFmk7g3AP_lsctw
def osdanmk7g3AP_lsctw_rec_ocr_routine( prototyper_name, feature_extractor_name, aam_name, seq_name,
                                 CAMname, pred_name, ctx_name, loss_name, part_loss_name, dom_loss_name, prec_loss_name, frec_loss_name,
                                 label_name, image_name,proto_name,target_name,tdict_name,length_name, log_path, log_each, name, maxT):

    dic={
        "maxT": maxT,
        "name":name,
        "routine":neko_HDOS2C_routine_CFmk7g3AP_lsctw,
        "mod_cvt_dicts":
        {
            "prototyper":prototyper_name,
            "feature_extractor":feature_extractor_name,
            "aam":aam_name,
            "CAM":CAMname,
            "seq": seq_name,
            "preds":pred_name,
            "losses":loss_name,
            "ctxmod": ctx_name,
            "dom_mix":dom_loss_name,
            "p_recon":prec_loss_name,
            "f_recon":frec_loss_name,
            "part_loss":part_loss_name
        },
        "inp_cvt_dicts":
        {
            "label":label_name,
            "image":image_name,
            "proto":proto_name,
            "target":target_name,
            "length":length_name,
            "tdict":tdict_name,
        },
        "log_path":log_path,
        "log_each":log_each,
    }

    return dic;