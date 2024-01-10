from neko_2021_mjt.modulars.dan.classifiers.neko_oslin import\
    neko_openset_linear_classifierK,neko_openset_linear_classifierKPM
from neko_sdk.MJT.default_config import get_default_model

def get_link_xos(arg_dict,path,optim_path=None):
    args={
    };
    return get_default_model(neko_openset_linear_classifierK,args,path,arg_dict["with_optim"],optim_path);

def config_linxos():
    return \
    {
        "save_each": 20000,
        "modular": get_link_xos,
        "args":
            {
                "with_optim": True
            },
    }
def get_linkPM_xos(arg_dict,path,optim_path=None):
    args={
    };
    return get_default_model(neko_openset_linear_classifierKPM,args,path,arg_dict["with_optim"],optim_path);

def config_linxosPM():
    return \
    {
        "save_each": 20000,
        "modular": get_linkPM_xos,
        "args":
            {
                "with_optim": True
            },
    }