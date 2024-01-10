from neko_2022_soai_zero.modules.spatial_attention_rect import \
    spatial_attention_rect
from neko_sdk.MJT.default_config import get_default_model


def get_sa_rect(arg_dict,path,optim_path=None):
    args={"w_parts":arg_dict["w_parts"],
          "h_parts":arg_dict["h_parts"],
          "detached":arg_dict["detached"]};
    return get_default_model(spatial_attention_rect,args,path,arg_dict["with_optim"],optim_path);

def config_sa_rect(h_parts=1,w_parts=1,detached=False):
    return \
    {
        "modular": get_sa_rect,
        "save_each": 20000,
        "args":
            {
                "with_optim": False,
                "w_parts":w_parts,
                "h_parts":h_parts,
                "detached":detached
            },
    }