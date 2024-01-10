from neko_2022_soai_zero.modules.spatial_attention_mk3spS import spatial_attention_mk3_seintr
from neko_sdk.MJT.default_config import get_default_model


def get_sa_mk3_seintr(arg_dict,path,optim_path=None):
    args={"ifc":arg_dict["num_channels"],
          "n_parts":arg_dict["n_parts"],
          "se_channel":arg_dict["se_channel"],
          "detached":arg_dict["detached"]};
    # args["num_channels"]=arg_dict["num_channels"]
    return get_default_model(spatial_attention_mk3_seintr,args,path,arg_dict["with_optim"],optim_path);

def config_sa_mk3_seintr(feat_ch=512,se_channel=32,n_parts=1,detached=False):
    return \
    {
        "modular": get_sa_mk3_seintr,
        "save_each": 20000,
        "args":
            {
                "with_optim": True,
                "n_parts":n_parts,
                "num_channels":feat_ch,
                "se_channel":se_channel,
                "detached":detached
            },
    }