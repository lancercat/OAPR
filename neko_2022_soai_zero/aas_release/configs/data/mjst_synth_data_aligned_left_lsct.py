

import os

from neko_2021_mjt.configs.data.mjst_data import get_test_all_uncased_dsrgb
from neko_2022_soai_zero.aas_release.configs.data.mjst_data_aligned_left import get_mjstcqa_AL, \
    get_eval_mjst_colorAL
from neko_2022_soai_zero.aas_release.data_cfg import arm_task
from neko_sdk.dataloaders.neko_joint_loader import neko_joint_loader;


def get_eval_dss_AL48_eng(dsroot,maxT_mjst,batch_size=1):
    te_meta_path =  os.path.join(dsroot, "dicts", "dab62cased.pt");
    get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, hw=[32, 128], batchsize=batch_size)
    mjst_eval_ds = get_eval_mjst_colorAL(dsroot, maxT_mjst,hw=[48,200],batch_size=batch_size);
    return te_meta_path,mjst_eval_ds,False

def get_dataloadercfgmjst48(root,tr_meta_path,maxT,bsize,armlsct):
    dscfg={
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            # "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            "dan_mjst": get_mjstcqa_AL(root, maxT, bsize, -1,hw=[48,200]),
            # The metas will try to avoid taking unseen testing characters in LSCT.
        }
    }
    
    return dscfg
def get_dssmjst_AL48_lsct(dsroot,maxT,bsize,armlsct=True):
    tr_meta_path=os.path.join(dsroot, "dicts", "dab62cased.pt");


    train_joint_ds=get_dataloadercfgmjst48(dsroot
                                          ,tr_meta_path,maxT,bsize,armlsct);
    return tr_meta_path,train_joint_ds
def get_close_set_taskAL48(dsroot,prefix,log_path,maxT):
    task_dict=arm_task({},get_eval_dss_AL48_eng,dsroot,prefix,log_path,maxT,"CLOSE-MJST");
    return task_dict;

