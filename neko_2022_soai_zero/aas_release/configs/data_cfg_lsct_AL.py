from neko_2022_soai_zero.aas_release.configs.data.chs_jpn_synth_data_aligned_left_lsct import \
    get_dssscht_AL48_lsct
from neko_2022_soai_zero.aas_release.configs.data.mjst_synth_data_aligned_left_lsct import \
    get_dssmjst_AL48_lsct, get_close_set_taskAL48
from neko_2022_soai_zero.aas_release.data_cfg_AL import get_gzsl_taskAL48_jk


def get_oss_training_protocolAL48_lsct(dsroot,tag,log_path,bsize=48,armlsct=True):
    maxT_chs=30;
    tr_meta_path_chs, train_joint_ds = get_dssscht_AL48_lsct(dsroot, maxT_chs, bsize,armlsct);


    # task_dict = arm_base_task_default2(task_dict, tag, osdanmk7_eval_routine_cfg, maxT_chs, te_meta_path_kr,
    #                                    kr_eval_ds,
    #                                    log_path,name="GZSL-CHS-KR");
    return train_joint_ds,tr_meta_path_chs,maxT_chs,get_gzsl_taskAL48_jk(dsroot, tag, log_path, maxT_chs);
def get_css_training_protocolAL48_lsct(dsroot,tag,log_path,bsize=48,armlsct=True):
    maxT_mjst=30;
    tr_meta_path_chs, train_joint_ds = get_dssmjst_AL48_lsct(dsroot, maxT_mjst, bsize,armlsct=armlsct);


    # task_dict = arm_base_task_default2(task_dict, tag, osdanmk7_eval_routine_cfg, maxT_chs, te_meta_path_kr,
    #                                    kr_eval_ds,
    #                                    log_path,name="GZSL-CHS-KR");
    return train_joint_ds,tr_meta_path_chs,maxT_mjst,get_close_set_taskAL48(dsroot,tag,log_path,maxT_mjst);
