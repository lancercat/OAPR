from neko_2022_soai_zero.modules.neko_bogo_softlink import neko_bogo_softlink

def config_softlink(model):
    return {
        "bogo_mod": neko_bogo_softlink,
        "args":
        {
            "model":model,
        }
    }