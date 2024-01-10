import os

from neko_2021_mjt.configs.data.chs_jpn_data import get_chs_HScqa, get_eval_jpn_color, \
    get_chs_tr_meta, get_chs_sc_meta, get_chs_mc_meta, \
    get_jpn_te_meta, get_eval_monkey_color, \
    get_jpn_te_meta_gosr, get_jpn_te_meta_ostr, get_jpn_te_meta_osr, \
    get_eval_kr_color, get_kr_te_meta, \
    get_eval_be_color, get_be_te_meta, \
    get_eval_hn_color, get_hn_te_meta
from neko_2021_mjt.configs.data.mjst_data import get_mjstcqa_cfg, get_test_all_uncased_dsrgb, \
    get_test_all_uncased_dsrgb_all
from neko_sdk.dataloaders.neko_joint_loader import neko_joint_loader


def get_dataloadercfgs(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize,devices=None):
    dss= {
        "loadertype": neko_joint_loader,
        "subsets":
        {
        }
    };
    if(maxT_mjst>0):
        # reset iterator gives deadlock, so we give a large enough repeat number
        dss["subsets"]["dan_mjst"]= get_mjstcqa_cfg(root, maxT_mjst, bs=bsize, hw=[32, 128]);
    if(maxT_chsHS>0):
        dss["subsets"]["dan_chs"] =  get_chs_HScqa(root, maxT_chsHS, bsize, -1, hw=[32, 128]);
    return dss;

def get_dataloadercfgsch(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            # "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            "dan_chs": get_chs_HScqa(root, maxT_chsHS, bsize, -1,hw=[32,128]),
        }
    }
def get_dataloadercfgsms(root,te_meta_path,tr_meta_path,maxT_mjst,maxT_chsHS,bsize):
    return \
    {
        "loadertype": neko_joint_loader,
        "subsets":
        {
            # reset iterator gives deadlock, so we give a large enough repeat number
            "dan_mjst":get_mjstcqa_cfg(root, maxT_mjst, bs=bsize,hw=[32,128]),
            #"dan_chs": get_chs_HScqa(root, maxT_chsHS, bsize, -1,hw=[32,128]),
        }
    }
def get_eval_dss(dsroot,maxT_mjst,maxT_chs,batch_size=1):
    te_meta_path_chsjpn = get_jpn_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None,hw=[32,128],batchsize=batch_size)
    chs_eval_ds = get_eval_jpn_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjpn,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds



def get_eval_dss_gosr(dsroot,maxT_mjst,maxT_chs):
    te_meta_path_chsjpn = get_jpn_te_meta_gosr(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
    chs_eval_ds = get_eval_jpn_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjpn,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds
def get_eval_dss_osr(dsroot, maxT_mjst, maxT_chs):
    te_meta_path_chsjpn = get_jpn_te_meta_osr(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
    chs_eval_ds = get_eval_jpn_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjpn,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds
def get_eval_dss_ostr(dsroot, maxT_mjst, maxT_chs):
    te_meta_path_chsjpn = get_jpn_te_meta_ostr(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
    chs_eval_ds = get_eval_jpn_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjpn,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds

def get_bench_dss_all(dsroot,maxT_mjst,maxT_chs,batch_size=1):
    te_meta_path_chsjpn = get_jpn_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb_all(maxT_mjst, dsroot, None, batch_size,hw=[32,128])
    chs_eval_ds = get_eval_jpn_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjpn,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds

def get_bench_dss(dsroot,maxT_mjst,maxT_chs,batch_size=1):
    te_meta_path_chsjpn = get_jpn_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, batch_size,hw=[32,128])
    chs_eval_ds = get_eval_jpn_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjpn,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds

def get_eval_dss_ba(dsroot,maxT_mjst,maxT_chs):
    te_meta_path_chsjpn = get_be_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
    chs_eval_ds = get_eval_be_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjpn,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds

def get_eval_dss_hn(dsroot,maxT_mjst,maxT_chs):
    te_meta_path_chsjpn = get_hn_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
    chs_eval_ds = get_eval_hn_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_chsjpn,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds


def get_dss(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjpn, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjpn = get_chs_tr_meta(dsroot);
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    train_joint_ds=get_dataloadercfgs(dsroot,te_meta_path_mjst,tr_meta_path_chsjpn,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjpn,te_meta_path_chsjpn,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds


def get_eval_dss_m(dsroot,maxT_mjst,maxT_chs,lang="chs"):
    te_meta_path_chsjpn = get_jpn_te_meta(dsroot);
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None, 16,hw=[32,128])
    jpnm_eval_ds = get_eval_monkey_color(dsroot, maxT_chs,lang,hw=[32,128]);

    return te_meta_path_chsjpn,te_meta_path_mjst,mjst_eval_ds,jpnm_eval_ds



def get_dsssc(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjpn, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjpn = get_chs_sc_meta(dsroot);
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    train_joint_ds=get_dataloadercfgs(dsroot,te_meta_path_mjst,tr_meta_path_chsjpn,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjpn,te_meta_path_chsjpn,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds

def get_dssmc(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjpn, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjpn = get_chs_mc_meta(dsroot);
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    train_joint_ds=get_dataloadercfgs(dsroot,te_meta_path_mjst,tr_meta_path_chsjpn,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjpn,te_meta_path_chsjpn,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds
def get_dsssch(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjpn, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjpn = get_chs_sc_meta(dsroot);
    tr_meta_path_mjst = None;
    train_joint_ds=get_dataloadercfgsch(dsroot,te_meta_path_mjst,tr_meta_path_chsjpn,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjpn,te_meta_path_chsjpn,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds

def get_dssscm(dsroot,maxT_mjst,maxT_chs,bsize):
    te_meta_path_chsjpn, te_meta_path_mjst, mjst_eval_ds, chs_eval_ds=\
        get_eval_dss(dsroot,maxT_mjst,maxT_chs);
    tr_meta_path_chsjpn =None;
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    train_joint_ds=get_dataloadercfgsms(dsroot,te_meta_path_mjst,tr_meta_path_chsjpn,maxT_mjst,maxT_chs,bsize);
    return tr_meta_path_chsjpn,te_meta_path_chsjpn,tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,chs_eval_ds,train_joint_ds



def get_dssscht(dsroot,maxT_chs,bsize):
    tr_meta_path_chsjpn = get_chs_sc_meta(dsroot);
    train_joint_ds=get_dataloadercfgsch(dsroot,None,None,None,maxT_chs,bsize);
    return tr_meta_path_chsjpn,train_joint_ds

def get_eval_dss_jpn(dsroot,maxT_chs,batch_size=1):
    te_meta_path_jpn = get_jpn_te_meta(dsroot);
    jpn_eval_ds = get_eval_jpn_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_jpn,jpn_eval_ds,False

def get_eval_dss_kr(dsroot,maxT_chs,batch_size=1):
    te_meta_path_kr = get_kr_te_meta(dsroot);
    kr_eval_ds = get_eval_kr_color(dsroot, maxT_chs,hw=[32,128]);
    return te_meta_path_kr,kr_eval_ds,False

def get_eval_dss_jk(dsroot,maxT_chs,batch_size=1):
    te_meta_path_kr, kr_eval_ds,_=get_eval_dss_kr(dsroot,maxT_chs,batch_size);
    te_meta_path_jpn, jpn_eval_ds,_=get_eval_dss_jpn(dsroot,maxT_chs,batch_size);
    return te_meta_path_jpn,te_meta_path_kr,jpn_eval_ds,kr_eval_ds

def get_eval_dss_close(dsroot,maxT_mjst,batch_size=1):
    te_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    mjst_eval_ds = get_test_all_uncased_dsrgb(maxT_mjst, dsroot, None,hw=[32,128],batchsize=batch_size)
    return te_meta_path_mjst,mjst_eval_ds



def get_dss_close(dsroot,maxT_mjst,bsize):
    te_meta_path_mjst, mjst_eval_ds=get_eval_dss_close(dsroot,maxT_mjst)
    tr_meta_path_mjst = os.path.join(dsroot, "dicts", "dab62cased.pt");
    train_joint_ds=get_dataloadercfgs(dsroot,te_meta_path_mjst,None,maxT_mjst,-1,bsize);
    return tr_meta_path_mjst,te_meta_path_mjst,mjst_eval_ds,train_joint_ds
