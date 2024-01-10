from neko_2021_mjt.configs.data.chs_jpn_data import get_jpn_te_meta,get_chs_sc_meta,get_kr_te_meta

from neko_2022_soai_zero.aas_release.configs.data.chs_jpn_data_aligned_left import get_chs_HScqa_AL
from neko_sdk.dataloaders.neko_joint_loader import neko_joint_loader;


def get_dataloadercfgsch48(root,te_meta_paths,tr_meta_path,maxT_mjst,maxT_chsHS,bsize,armlsct):
    dsscfg={
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            # "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            "dan_chs": get_chs_HScqa_AL(root, maxT_chsHS, bsize, -1,hw=[48,200]),
            # The metas will try to avoid taking unseen testing characters in LSCT.
        }
    };

    return dsscfg

def get_dssscht_AL48_lsct(dsroot,maxT_chs,bsize,armlsct=False):
    tr_meta_path_chsjpn = get_chs_sc_meta(dsroot);
    # again we need to remove all unseen characters from LSCT---
    # to make sure it's unseen, this is for benchmarking proposes.
    # In really life, we don't care. If LSCT hits, it's even better than not.
    # But for science, we want to estimation the performance on characters that are not covered.
    # Thus, we remove them from synthetic data during benchmarking, for estimation proposes.
    train_joint_ds=get_dataloadercfgsch48(dsroot,[get_jpn_te_meta(dsroot),
                                                  get_kr_te_meta(dsroot)]
                                          ,tr_meta_path_chsjpn,None,maxT_chs,bsize,armlsct);
    return tr_meta_path_chsjpn,train_joint_ds



