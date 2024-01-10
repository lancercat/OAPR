import os
from functools import partial

from configs import model_mod_cfg as modcfg
from neko_2022_soai_zero.project_aas_mk3_lsct.configs.data_cfg_lsct_AL import get_close_set_taskAL48


def dan_eval_cfg(evalfn,tag,
        save_root,dsroot,
        log_path,iterkey,prefix="base_mjst_"):

    if(log_path):
        epath=os.path.join(log_path, tag);
    else:
        epath=None;
    task_dict=evalfn(dsroot,prefix,log_path,30);

    return \
    {
        "root": save_root,
        "iterkey": iterkey, # something makes no sense to start fresh
        "modules": modcfg(None,None,25,30),
        "export_path":epath,
        "tasks":task_dict
    }

dan_open_all={
    # "OSR": partial(dan_mjst_eval_cfg, get_eval_dss_osr, "OSR"),
    "Close":partial(dan_eval_cfg, get_close_set_taskAL48, "Close"),
    # "GOSR": partial(dan_mjst_eval_cfg, get_eval_dss_gosr,"GOSR"),
    # "OSTR": partial(dan_mjst_eval_cfg, get_eval_dss_ostr,"OSTR"),

}
