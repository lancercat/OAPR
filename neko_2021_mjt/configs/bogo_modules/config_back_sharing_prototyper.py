from neko_sdk.MJT.bogo_module.prototype_gen3 import prototyper_gen3,prototyper_gen3p,prototyper_gen3pe,prototyper_gen3_masked,prototyper_gen3peT;
def config_prototyper_gen3d(sp_proto, backbone, cam, drop=None, capacity=512, force_proto_shape=None):
    return {
        "bogo_mod": prototyper_gen3,
        "args":
        {
            "detached_ga": True,
            "capacity":capacity,
            "sp_proto":sp_proto,
            "backbone":backbone,
            "cam":cam,
            "drop":drop,
            "force_proto_shape":force_proto_shape,
        }
    }
def config_prototyper_gen3(sp_proto, backbone, cam, drop=None, capacity=512, force_proto_shape=None,detached_ga=False):
    return {
        "bogo_mod": prototyper_gen3,
        "args":
        {
            "capacity":capacity,
            "sp_proto":sp_proto,
            "backbone":backbone,
            "cam":cam,
            "drop":drop,
            "force_proto_shape":force_proto_shape,
            "detached_ga":detached_ga,
        }
    }
def config_prototyper_gen3p(sp_proto, backbone, cam, drop=None, capacity=512, force_proto_shape=None,n_part=1,detached_ga=False):
    return {
        "bogo_mod": prototyper_gen3p,
        "args":
        {
            "capacity":capacity,
            "sp_proto":sp_proto,
            "backbone":backbone,
            "cam":cam,
            "drop":drop,
            "force_proto_shape":force_proto_shape,
            "detached_ga": detached_ga
        }
    }
def config_prototyper_gen3pe(sp_proto, backbone, cam, drop=None, capacity=512, force_proto_shape=None,n_part=1,detached_ga=False):
    return {
        "bogo_mod": prototyper_gen3pe,
        "args":
        {
            "capacity":capacity,
            "sp_proto":sp_proto,
            "backbone":backbone,
            "cam":cam,
            "drop":drop,
            "force_proto_shape":force_proto_shape,
            "detached_ga": detached_ga
        }
    }

def config_prototyper_gen3peT(sp_proto, backbone, cam, drop=None, capacity=512, force_proto_shape=None,n_part=1,detached_ga=False):
    return {
        "bogo_mod": prototyper_gen3peT,
        "args":
        {
            "capacity":capacity,
            "sp_proto":sp_proto,
            "backbone":backbone,
            "cam":cam,
            "drop":drop,
            "force_proto_shape":force_proto_shape,
            "detached_ga": detached_ga
        }
    }
def config_prototyper_gen3_masked(sp_proto, backbone, cam, drop=None, capacity=512, force_proto_shape=None,detached_ga=False):
    return {
        "bogo_mod": prototyper_gen3_masked,
        "args":
        {
            "capacity":capacity,
            "sp_proto":sp_proto,
            "backbone":backbone,
            "cam":cam,
            "drop":drop,
            "force_proto_shape":force_proto_shape,
            "detached_ga":detached_ga
        }
    }
