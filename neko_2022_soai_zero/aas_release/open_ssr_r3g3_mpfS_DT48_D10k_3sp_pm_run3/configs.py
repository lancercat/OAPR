
from neko_2022_soai_zero.configs.routines.arm_aas_routine import arm_aas_routine
from neko_2022_soai_zero.configs.routines.osdanmk7g3ap_routine_cfg import osdanmk7g3APD_rec_ocr_routine
from neko_2022_soai_zero.aas_release.configs.data_cfg_lsct_AL import get_oss_training_protocolAL48_lsct
from neko_2022_soai_zero.aas_release.configs.loadouts.aas_mk3_lsct_ss import \
    arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7g2_mpfS_3sp_pm_covl_dtcg3_32

TAG="base_chs_"

def model_mod_cfg(tr_meta_path_chs,tr_meta_path_mjst,maxT_mjst,maxT_chs):
    capacity=256;
    feat_ch=512;
    mods={};
    mods=arm_module_set_r45trinorm_orig_dsa3hGTAnp_mk7g2_mpfS_3sp_pm_covl_dtcg3_32(
        mods,"base_chs_",maxT_chs,capacity,feat_ch,tr_meta_path_chs,wemb=0,wpart=3);
    return mods;



def dan_single_model_train_cfg(save_root,dsroot,
                               log_path,log_each,itrk= "Top Nep",bsize=48,tvitr=200000):
    maxT_chs=30;
    maxT_mjst=25;
    train_joint_ds,tr_meta_path_chs,maxT_chs,task_dict=get_oss_training_protocolAL48_lsct(dsroot,tag="base_chs_",log_path=log_path,armlsct=False)


    routines = {};
    # routines = arm_base_routine2(routines, "base_mjst_", osdanmk7dt_ocr_routine, maxT_mjst, log_path,
    #                             log_each, "dan_mjst_");
    routines = arm_aas_routine(routines, "base_chs_", osdanmk7g3APD_rec_ocr_routine, maxT_chs, log_path,
                               log_each, "dan_chs_",delay_till=10*1000);

    return \
        {
            "root": save_root,
            "val_each": 10000,
            "vitr": 200000,
            "vepoch": 2,
            "iterkey": itrk,  # something makes no sense to start fresh
            "dataloader_cfg":train_joint_ds,
            # make sure the unseen characters are unseen.
            "modules": model_mod_cfg(tr_meta_path_chs,None, maxT_mjst,maxT_chs),
            "routine_cfgs": routines,
            "tasks": task_dict,
        }