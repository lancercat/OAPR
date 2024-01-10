
from neko_2020nocr.dan.configs.pipelines_pami import get_cam_args
from neko_2022_soai_zero.modules.cam_stop_se import neko_CAM_stop_seintr,neko_CAM_stop_mpf_seintr,neko_CAM_stop_mp_seintr,neko_CAM_stop_mpfl_seintr

from neko_sdk.MJT.default_config import get_default_model

def get_cam_stop_seintr(arg_dict,path,optim_path=None):
    args=get_cam_args(arg_dict["maxT"],arg_dict["cam_ch"]);
    args["scales"]=arg_dict["scales"];
    args["detached"]=arg_dict["detached"];
    args["num_se_channels"]=arg_dict["num_se_channels"];
    # args["num_channels"]=arg_dict["num_channels"]
    return get_default_model(neko_CAM_stop_seintr,args,path,arg_dict["with_optim"],optim_path);

def config_cam_stop_seintr(maxT,scales=None,feat_ch=512,expf=1,cam_ch=64,num_se_channels=32,detached=False):
    if scales is None:
        scales=[
                    [int(expf*64+num_se_channels), 16, 64],
                    [int(expf*256+num_se_channels), 8, 32],
                    [int(feat_ch+num_se_channels), 8, 32]
                ];
    print(scales);
    return \
    {
        "modular": get_cam_stop_seintr,
        "save_each": 20000,
        "args":
            {
                "detached":detached,
                "cam_ch":cam_ch,
                "num_channels":feat_ch,
                "num_se_channels":num_se_channels,
                "scales": scales,
                "maxT": maxT,
                "with_optim": True
            },
    }



def get_cam_stop_mp_seintr(arg_dict,path,optim_path=None):
    args=get_cam_args(arg_dict["maxT"],arg_dict["cam_ch"]);
    args["scales"]=arg_dict["scales"];
    args["n_parts"]=arg_dict["n_parts"];
    args["num_se_channels"]=arg_dict["num_se_channels"];

    # args["num_channels"]=arg_dict["num_channels"]
    return get_default_model(neko_CAM_stop_mp_seintr,args,path,arg_dict["with_optim"],optim_path);

def config_cam_stop_mp_seintr(maxT,scales=None,feat_ch=512,expf=1,cam_ch=64,num_se_channels=32,n_parts=4):
    if scales is None:
        scales=[
                    [int(expf*64)+num_se_channels, 16, 64],
                    [int(expf*256)+num_se_channels, 8, 32],
                    [int(feat_ch)+num_se_channels, 8, 32]
                ];
    print(scales);
    return \
    {
        "modular": get_cam_stop_mp_seintr,
        "save_each": 20000,
        "args":
            {
                "n_parts":n_parts,
                "cam_ch":cam_ch,
                "num_channels":feat_ch,
                "num_se_channels": num_se_channels,
                "scales": scales,
                "maxT": maxT,
                "with_optim": True
            },
    }


def get_cam_stop_mpf_seintr(arg_dict,path,optim_path=None):
    args=get_cam_args(arg_dict["maxT"],arg_dict["cam_ch"]);
    args["scales"]=arg_dict["scales"];
    args["n_parts"]=arg_dict["n_parts"];
    args["num_se_channels"]=arg_dict["num_se_channels"];
    args["detached"]=arg_dict["detached"];
    # args["num_channels"]=arg_dict["num_channels"]
    return get_default_model(neko_CAM_stop_mpf_seintr,args,path,arg_dict["with_optim"],optim_path);

def config_cam_stop_mpf_seintr(maxT,scales=None,feat_ch=512,expf=1,cam_ch=64,num_se_channels=32,n_parts=4,detached=False):
    if scales is None:
        scales=[
                    [int(expf*32)+num_se_channels, 16, 64],
                    [int(expf*128)+num_se_channels, 8, 32],
                    [int(feat_ch)+num_se_channels, 8, 32]
                ];
    print(scales);
    return \
    {
        "modular": get_cam_stop_mpf_seintr,
        "save_each": 20000,
        "args":
            {
                "n_parts":n_parts,
                "cam_ch":cam_ch,
                "num_channels":feat_ch,
                "num_se_channels": num_se_channels,
                "scales": scales,
                "maxT": maxT,
                "detached":detached,
                "with_optim": True
            },
    }


def get_cam_stop_mpfl_seintr(arg_dict,path,optim_path=None):
    args=get_cam_args(arg_dict["maxT"],arg_dict["cam_ch"]);
    args["scales"]=arg_dict["scales"];
    args["n_parts"]=arg_dict["n_parts"];
    args["num_se_channels"]=arg_dict["num_se_channels"];
    args["detached"]=arg_dict["detached"];
    # args["num_channels"]=arg_dict["num_channels"]
    return get_default_model(neko_CAM_stop_mpfl_seintr,args,path,arg_dict["with_optim"],optim_path);

def config_cam_stop_mpfl_seintr(maxT,scales=None,feat_ch=512,expf=1,cam_ch=64,num_se_channels=32,n_parts=4,detached=False):
    if scales is None:
        scales=[
                    [int(expf*32)+num_se_channels, 16, 64],
                    [int(expf*128)+num_se_channels, 8, 32],
                    [int(feat_ch)+num_se_channels, 8, 32]
                ];
    print(scales);
    return \
    {
        "modular": get_cam_stop_mpfl_seintr,
        "save_each": 20000,
        "args":
            {
                "n_parts":n_parts,
                "cam_ch":cam_ch,
                "num_channels":feat_ch,
                "num_se_channels": num_se_channels,
                "scales": scales,
                "maxT": maxT,
                "detached":detached,
                "with_optim": True
            },
    }
