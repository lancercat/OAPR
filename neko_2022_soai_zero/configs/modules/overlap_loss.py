from neko_2022_soai_zero.modules.neko_part_loss import part_overlap_loss,att_distribute_loss_dtcg3
def get_overlap_loss(arg_dict,path,optim_path=None):
    mod=part_overlap_loss(arg_dict);
    return mod,None,None;
def config_overlap_loss():
    return \
    {
        "save_each": 0,
        "modular": get_overlap_loss,
        "args":{},
    }

def get_variance_dtcg3_loss(arg_dict,path,optim_path=None):
    mod=att_distribute_loss_dtcg3(arg_dict);
    return mod,None,None;
def config_variance_dtcg3_loss(weight=1):
    return \
        {
            "save_each":0,
            "modular":get_variance_dtcg3_loss,
            "args":{"weight":weight},
        }