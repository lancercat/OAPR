from neko_2022_soai_zero.aas_release.configs.data.dual_no_lsct32_AL import \
    get_eval_dss_jpn_AL,get_eval_dss_jpn_AL_jk,get_dssscht_AL,get_eval_dss_jpn_AL_osr,get_eval_dss_jpn_AL_gosr,get_eval_dss_jpn_AL_ostr, \
    get_eval_dss_jk_AL48,get_eval_dss_jpn_ostr_AL48,get_eval_dss_jpn_gosr_AL48,get_eval_dss_jpn_gzsl_AL48,get_eval_dss_jpn_osr_AL48
from neko_2022_soai_zero.aas_release.data_cfg import arm_task,arm_task_part,arm_task_part_T


def get_gzsl_taskAL(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task({},get_eval_dss_jpn_AL,dsroot,prefix,log_path,maxT_chs,"GZSL-CHS-JP");
    return task_dict;
def get_osr_taskAL(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task({},get_eval_dss_jpn_AL_osr,dsroot,prefix,log_path,maxT_chs,"OSR-CHS-JP");
    return task_dict;

def get_gosr_taskAL(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task({},get_eval_dss_jpn_AL_gosr,dsroot,prefix,log_path,maxT_chs,"GOSR-CHS-JP");
    return task_dict;

def get_ostr_taskAL(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task({},get_eval_dss_jpn_AL_ostr,dsroot,prefix,log_path,maxT_chs,"OSTR-CHS-JP");
    return task_dict;



def get_gzsl_taskAL_jk(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task({},get_eval_dss_jpn_AL_jk,dsroot,prefix,log_path,maxT_chs,"GZSL-CHS-JP");
    return task_dict;

def get_gzsl_task_partAL(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task_part({},get_eval_dss_jpn_AL,dsroot,prefix,log_path,maxT_chs,"GZSL-CHS-JP");
    return task_dict;
def get_gzsl_task_partAL_jk(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task_part({},get_eval_dss_jpn_AL_jk,dsroot,prefix,log_path,maxT_chs,"GZSL-CHS-JP");
    return task_dict;

def get_gzsl_task_partALT(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task_part_T({},get_eval_dss_jpn_AL,dsroot,prefix,log_path,maxT_chs,"GZSL-CHS-JP");
    return task_dict;
def get_oss_training_protocolAL(dsroot,tag,log_path,bsize=48):
    maxT_chs=30;
    tr_meta_path_chs, train_joint_ds = get_dssscht_AL(dsroot, maxT_chs, bsize);


    # task_dict = arm_base_task_default2(task_dict, tag, osdanmk7_eval_routine_cfg, maxT_chs, te_meta_path_kr,
    #                                    kr_eval_ds,
    #                                    log_path,name="GZSL-CHS-KR");
    return train_joint_ds,tr_meta_path_chs,maxT_chs,get_gzsl_taskAL(dsroot,tag,log_path,maxT_chs);

def get_oss_training_protocol_partAL(dsroot,tag,log_path,bsize=48):
    maxT_chs=30;
    tr_meta_path_chs, train_joint_ds = get_dssscht_AL(dsroot, maxT_chs, bsize);
    return train_joint_ds,tr_meta_path_chs,maxT_chs,get_gzsl_task_partAL(dsroot,tag,log_path,maxT_chs);
def get_oss_training_protocol_partALmt(dsroot,tag,log_path,bsize=48):
    maxT_chs=30;
    tr_meta_path_chs, train_joint_ds = get_dssscht_AL(dsroot, maxT_chs, bsize);
    return train_joint_ds,tr_meta_path_chs,maxT_chs,get_gzsl_task_partAL(dsroot,tag,log_path,maxT_chs);

def get_oss_training_protocol_partALT(dsroot,tag,log_path,bsize=48):
    maxT_chs=30;
    tr_meta_path_chs, train_joint_ds = get_dssscht_AL(dsroot, maxT_chs, bsize);
    return train_joint_ds,tr_meta_path_chs,maxT_chs,get_gzsl_task_partALT(dsroot,tag,log_path,maxT_chs);



def get_gzsl_taskAL48_jk(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task({}, get_eval_dss_jk_AL48, dsroot, prefix, log_path, maxT_chs, "GZSL-CHS-JP-KR");
    return task_dict;

def get_gzsl_taskAL48_jpn(dsroot, prefix, log_path, maxT_chs):
    task_dict=arm_task({}, get_eval_dss_jpn_gzsl_AL48, dsroot, prefix, log_path, maxT_chs, "GZSL-CHS-JP");
    return task_dict;
def get_osr_taskAL48_jpn(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task({},get_eval_dss_jpn_osr_AL48,dsroot,prefix,log_path,maxT_chs,"OSR-CHS-JP");
    return task_dict;

def get_gosr_taskAL48_jpn(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task({},get_eval_dss_jpn_gosr_AL48,dsroot,prefix,log_path,maxT_chs,"GOSR-CHS-JP");
    return task_dict;

def get_ostr_taskAL48_jpn(dsroot,prefix,log_path,maxT_chs):
    task_dict=arm_task({},get_eval_dss_jpn_ostr_AL48,dsroot,prefix,log_path,maxT_chs,"OSTR-CHS-JP");
    return task_dict;