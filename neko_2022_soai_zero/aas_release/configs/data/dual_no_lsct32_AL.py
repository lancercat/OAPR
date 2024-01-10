from neko_2021_mjt.configs.data.chs_jpn_data import get_jpn_te_meta,get_jpn_te_meta_gosr,get_jpn_te_meta_osr,get_jpn_te_meta_ostr,get_chs_sc_meta,get_kr_te_meta

from neko_2022_soai_zero.aas_release.configs.data.chs_jpn_data_aligned_left import get_eval_jpn_colorAL,\
    get_chs_HScqa_AL,get_eval_kr_colorAL
from neko_sdk.dataloaders.neko_joint_loader import neko_joint_loader

def get_eval_dss_jpn_AL(dsroot,maxT_chs,batch_size=1):
    te_meta_path_jpn = get_jpn_te_meta(dsroot);
    jpn_eval_ds = get_eval_jpn_colorAL(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_jpn,jpn_eval_ds,False

def get_eval_dss_jpn_AL_osr(dsroot,maxT_chs,batch_size=1):
    te_meta_path_jpn = get_jpn_te_meta_osr(dsroot);
    jpn_eval_ds = get_eval_jpn_colorAL(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_jpn,jpn_eval_ds,True

def get_eval_dss_jpn_AL_gosr(dsroot,maxT_chs,batch_size=1):
    te_meta_path_jpn = get_jpn_te_meta_gosr(dsroot);
    jpn_eval_ds = get_eval_jpn_colorAL(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_jpn,jpn_eval_ds,True
def get_eval_dss_jpn_AL_ostr(dsroot,maxT_chs,batch_size=1):
    te_meta_path_jpn = get_jpn_te_meta_ostr(dsroot);
    jpn_eval_ds = get_eval_jpn_colorAL(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_jpn,jpn_eval_ds,True
def get_eval_dss_jpn_AL_jk(dsroot,maxT_chs,batch_size=1):
    jpn_eval_ds = get_eval_jpn_colorAL(dsroot, maxT_chs,hw=[32,128]);
    te_meta_path_jpn = get_jpn_te_meta(dsroot);

    te_meta_path_kr = get_kr_te_meta(dsroot);
    kr_eval_ds = get_eval_kr_colorAL(dsroot, maxT_chs,hw=[32,128]);

    return {"JAP":te_meta_path_jpn,"KR":te_meta_path_kr}, \
           {"JAP":jpn_eval_ds,"KR":kr_eval_ds},False

def get_eval_dss_jpn_gzsl_AL48(dsroot, maxT_chs, batch_size=1):
    te_meta_path_jpn = get_jpn_te_meta(dsroot);
    jpn_eval_ds = get_eval_jpn_colorAL(dsroot, maxT_chs,hw=[48,200]);
    return te_meta_path_jpn,jpn_eval_ds,False
def get_eval_dss_jpn_osr_AL48(dsroot, maxT_chs, batch_size=1):
    te_meta_path_jpn = get_jpn_te_meta_osr(dsroot);
    jpn_eval_ds = get_eval_jpn_colorAL(dsroot, maxT_chs,hw=[48,200]);
    return te_meta_path_jpn,jpn_eval_ds,True
def get_eval_dss_jpn_gosr_AL48(dsroot, maxT_chs, batch_size=1):
    te_meta_path_jpn = get_jpn_te_meta_gosr(dsroot);
    jpn_eval_ds = get_eval_jpn_colorAL(dsroot, maxT_chs,hw=[48,200]);
    return te_meta_path_jpn,jpn_eval_ds,True
def get_eval_dss_jpn_ostr_AL48(dsroot, maxT_chs, batch_size=1):
    te_meta_path_jpn = get_jpn_te_meta_ostr(dsroot);
    jpn_eval_ds = get_eval_jpn_colorAL(dsroot, maxT_chs,hw=[48,200]);
    return te_meta_path_jpn,jpn_eval_ds,True

def get_eval_dss_kr_AL48(dsroot, maxT_chs, batch_size=1):
    te_meta_path_kr = get_kr_te_meta(dsroot);
    kr_eval_ds = get_eval_kr_colorAL(dsroot, maxT_chs, hw=[48, 200]);
    return te_meta_path_kr,kr_eval_ds,False
def get_eval_dss_jk_AL48(dsroot, maxT_chs, batch_size=1):
    jpn_eval_ds = get_eval_jpn_colorAL(dsroot, maxT_chs,hw=[48,200]);
    te_meta_path_jpn = get_jpn_te_meta(dsroot);

    te_meta_path_kr = get_kr_te_meta(dsroot);
    kr_eval_ds = get_eval_kr_colorAL(dsroot, maxT_chs,hw=[48,200]);

    return {"JAP":te_meta_path_jpn,"KR":te_meta_path_kr}, \
           {"JAP":jpn_eval_ds,"KR":kr_eval_ds},False


def get_dataloadercfgsch(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            # "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            "dan_chs": get_chs_HScqa_AL(root, maxT_chsHS, bsize, -1,hw=[32,128]),
        }
    }
def get_dssscht_AL(dsroot,maxT_chs,bsize):
    tr_meta_path_chsjpn = get_chs_sc_meta(dsroot);
    train_joint_ds=get_dataloadercfgsch(dsroot,None,None,None,maxT_chs,bsize);
    return tr_meta_path_chsjpn,train_joint_ds
