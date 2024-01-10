import os
def dan_eval_cfg_common(modules,evalfn,tag,
        save_root,dsroot,
        log_path,iterkey,prefix="base_chs_"):

    if(log_path):
        epath=os.path.join(log_path, tag);
    else:
        epath=None;
    task_dict=evalfn(dsroot,prefix,log_path,30);

    return \
    {
        "root": save_root,
        "iterkey": iterkey, # something makes no sense to start fresh
        "modules": modules,
        "export_path":epath,
        "tasks":task_dict,
    }
