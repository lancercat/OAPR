from osocrNG.bogomods_g2.res45g2_bws.res45_g2_ffn_naive import neko_res45_bogo_g2_ffn_bws


def config_bogo_resbinorm_g2_core(conv_container,bn_container,engine):
    return {
        "bogo_mod": engine,
        "args":
        {
            "mod_cvt":
            {
                "conv":conv_container,
                "norm":bn_container,
            },
        }
    }


def config_bogo_resbinorm_g2_ffn(conv_container,bn_container,ffn_container):
    return {
        "bogo_mod": neko_res45_bogo_g2_ffn_bws,
        "args":
        {
            "mod_cvt":
            {
                "conv":conv_container,
                "norm":bn_container,
                "ffn":ffn_container,
            },
        }
    }