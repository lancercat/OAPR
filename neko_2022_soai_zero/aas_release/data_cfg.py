from neko_2021_mjt.configs.loadouts.base_module_set import arm_base_task_default2
from neko_2021_mjt.configs.routines.ocr_routines.mk7.osdanmk7_routine_cfg import \
    osdanmk7_eval_routine_cfg, osdanmk7P_eval_routine_cfg, osdanmk7PT_eval_routine_cfg
from neko_2021_mjt.dss_presets.dual_no_lsct_32 import get_dssscht, get_eval_dss_jpn, get_eval_dss_jk;


def arm_any_task(task_dict,eval_fn,routine_cfg,dsroot,prefix,log_path,maxT,name):
    te_meta_path, eval_ds,has_rej=eval_fn(dsroot,maxT);
    if(type(te_meta_path) is dict):
        for k in te_meta_path:
            task_dict = arm_base_task_default2(task_dict, prefix, routine_cfg, maxT, te_meta_path[k],
                                               eval_ds[k],
                                               log_path, name=name+"_"+k, measure_rej=has_rej);
    else:
        task_dict = arm_base_task_default2(task_dict, prefix, routine_cfg, maxT, te_meta_path,
                                           eval_ds,
                                           log_path, name=name,measure_rej=has_rej);
    return task_dict;

def arm_task(task_dict,eval_fn,dsroot,prefix,log_path,maxT,name,has_rej=False):
    return arm_any_task(task_dict,eval_fn,osdanmk7_eval_routine_cfg,dsroot,prefix,log_path,maxT,name);
def arm_task_part(task_dict,eval_fn,dsroot,prefix,log_path,maxT,name):
    return arm_any_task(task_dict,eval_fn,osdanmk7P_eval_routine_cfg,dsroot,prefix,log_path,maxT,name);

def arm_task_part_T(task_dict,eval_fn,dsroot,prefix,log_path,maxT,name):
    return arm_any_task(task_dict,eval_fn,osdanmk7PT_eval_routine_cfg,dsroot,prefix,log_path,maxT,name);

def get_gzsl_task(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task({},get_eval_dss_jpn,dsroot,prefix,log_path,maxT_chs,"GZSL-CHS-JP");
    return task_dict;

def get_gzsl_task_part(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task_part({},get_eval_dss_jpn,dsroot,prefix,log_path,maxT_chs,"GZSL-CHS-JP");
    return task_dict;
def get_gzsl_task_jk(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task({},get_eval_dss_jk,dsroot,prefix,log_path,maxT_chs,"GZSL-CHS-JP");
    return task_dict;

def get_gzsl_task_part_jk(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task_part({},get_eval_dss_jk,dsroot,prefix,log_path,maxT_chs,"GZSL-CHS-JP");
    return task_dict;



def get_oss_training_protocol(dsroot,tag,log_path,bsize=48):
    maxT_chs=30;
    tr_meta_path_chs, train_joint_ds = get_dssscht(dsroot, maxT_chs, bsize);


    # task_dict = arm_base_task_default2(task_dict, tag, osdanmk7_eval_routine_cfg, maxT_chs, te_meta_path_kr,
    #                                    kr_eval_ds,
    #                                    log_path,name="GZSL-CHS-KR");
    return train_joint_ds,tr_meta_path_chs,maxT_chs,get_gzsl_task(dsroot,tag,log_path,maxT_chs);

def get_oss_training_protocol_part(dsroot,tag,log_path,bsize=48):
    maxT_chs=30;
    tr_meta_path_chs, train_joint_ds = get_dssscht(dsroot, maxT_chs, bsize);


    return train_joint_ds,tr_meta_path_chs,maxT_chs,get_gzsl_task_part(dsroot,tag,log_path,maxT_chs);