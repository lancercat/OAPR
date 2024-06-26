from neko_2021_mjt.modulars.dan.neko_CFDTD_mk5 import neko_os_CFDTD_mk5,neko_os_CFDTD_mk5d,neko_os_CFDTD_mk5c,neko_os_CFDTD_mk5ce
from neko_sdk.MJT.default_config import get_default_model

def get_dtdmk5_xos(arg_dict,path,optim_path=None):
    args={
    };
    return get_default_model(neko_os_CFDTD_mk5,args,path,arg_dict["with_optim"],optim_path);

def config_dtdmk5():
    return \
    {
        "save_each": 20000,
        "modular": get_dtdmk5_xos,
        "args":
            {
                "with_optim": False
            },
    }
def get_dtdmk5c_xos(arg_dict,path,optim_path=None):
    args={
    };
    return get_default_model(neko_os_CFDTD_mk5c,args,path,arg_dict["with_optim"],optim_path);

def config_dtdmk5c():
    return \
    {
        "save_each": 20000,
        "modular": get_dtdmk5c_xos,
        "args":
            {
                "with_optim": False
            },
    }

def get_dtdmk5ce_xos(arg_dict,path,optim_path=None):
    args={
    };
    return get_default_model(neko_os_CFDTD_mk5ce,args,path,arg_dict["with_optim"],optim_path);

def config_dtdmk5ce():
    return \
    {
        "save_each": 20000,
        "modular": get_dtdmk5ce_xos,
        "args":
            {
                "with_optim": False
            },
    }



def get_dtdmk5d_xos(arg_dict,path,optim_path=None):
    args={
    };
    return get_default_model(neko_os_CFDTD_mk5d,args,path,arg_dict["with_optim"],optim_path);

def config_dtdmk5d():
    return \
    {
        "save_each": 20000,
        "modular": get_dtdmk5d_xos,
        "args":
            {
                "with_optim": False
            },
    }