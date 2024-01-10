from neko_sdk.MJT.eval_tasks.dan_eval_tasks import neko_odan_eval_tasks

def arm_base_eval_routine2(srcdst,tname,prefix,routine_type,log_path,maxT,measure_rej=False,override_dict=None):
    kvargs = {"prototyper_name": prefix + "prototyper",
              "sampler_name": prefix + "Latin_62_sampler",
              "feature_extractor_name": prefix + "feature_extractor_cco",
              "CAMname": prefix + "TA",
              "seq_name": prefix + "DTD",
              "pred_name": [prefix + "pred"],
              "loss_name": [prefix + "loss_cls_emb"],
              "image_name": "image",
              "label_name": "label",
              "log_path": log_path,
              "name": prefix + tname,
              "maxT": maxT,
              "measure_rej": measure_rej, }
    if(override_dict is not None):
        for k in override_dict:
            kvargs[k]=override_dict[k];
    return routine_type(
        **kvargs
    );

def arm_base_task_default2(srcdst,prefix,routine_type,maxT,te_meta_path,datasets,log_path,name="close_set_benchmarks",measure_rej=False,override_dict=None):
    te_routine={};
    te_routine=arm_base_eval_routine2(te_routine,"close_set_benchmark",
                                      prefix,routine_type,log_path,maxT,measure_rej=measure_rej,override_dict=override_dict)
    if(override_dict is not None):
        if("prototyper_name" in override_dict):
            pengine=override_dict["prototyper_name"];
        else:
            pengine=prefix + "prototyper"
    else:
        pengine = prefix + "prototyper"

    srcdst[prefix+name]={
        "type": neko_odan_eval_tasks,
        "protoname":pengine,
        "temeta":
            {
                "meta_path": te_meta_path,
                "case_sensitive": False,
            },
        "datasets":datasets,
        "routine_cfgs": te_routine,
    }
    return srcdst

