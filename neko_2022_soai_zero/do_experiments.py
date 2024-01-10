import os

from neko_sdk.ocr_modules.logparsers.compare_results_by_id import compare_files


def do_experiments(evalcfg,root,method,epoch="_E0",dev="318prirC",tag="base_",export_path="/home/lasercat/ssddata/export/"):
    from neko_sdk.MJT.lanuch_std_test import launchtest
    argv = ["Meeeeooooowwww",
            os.path.join(root,dev,method,"jtrmodels"),
            epoch,
            os.path.join(root,dev,method,"jtrmodels"),
            ]
    rawpath=os.path.join(root,dev,method,"jtrmodels/closeset_benchmarks/",tag+"chs_prototyper/JAP_lang/");
    os.makedirs(rawpath,exist_ok=True);
    launchtest(argv, evalcfg,export_path=export_path);
    compare_files(os.path.join(root,dev),[method],os.path.join(root,dev,method,"dashboard"),4009,[tag]);

def do_experiments2(evalcfgs,root,method,epoch="_E0",dev="318prirC",tag="base_",export_path="/run/media/lasercat/ssddata/export/GZSL/base_chs_prototyper/JAP_lang/",vdbg=None):
    from neko_sdk.MJT.lanuch_std_test import launchtest
    argv = ["Meeeeooooowwww",
            os.path.join(root,dev,method,"jtrmodels"),
            epoch,
            os.path.join(root,dev,method,"jtrmodels"),
            ]
    for k in evalcfgs:
        #                                                 protocol     MISD model          Dataset
        rawpath=os.path.join(root,dev,method,"jtrmodels/",k,      tag+"chs_prototyper", "JAP_lang/");
        os.makedirs(rawpath,exist_ok=True);
        if(export_path=="NEP_rawpath_NEP"):
            export_path=os.path.join(root,dev,method);
        launchtest(argv, evalcfgs[k],export_path=export_path,vdbg=vdbg);
        try:
            compare_files(os.path.join(root,dev),[method],os.path.join(root,dev,method,"dashboard"),4009,[tag]);
        except:
            print("no dashboard available")