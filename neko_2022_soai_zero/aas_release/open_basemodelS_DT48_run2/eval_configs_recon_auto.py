from functools import partial

from configs import model_mod_cfg as modcfg
from neko_2022_soai_zero.aas_release.test_protocol import dan_eval_cfg_common
from neko_2022_soai_zero.aas_release.data_cfg_AL import get_gzsl_taskAL48_jk, get_osr_taskAL48_jpn, \
    get_gosr_taskAL48_jpn, get_ostr_taskAL48_jpn

dan_open_all={
    "OSR": partial(dan_eval_cfg_common,modcfg(None,None,25,30),get_osr_taskAL48_jpn, "OSR"),
    "GZSL":partial(dan_eval_cfg_common,modcfg(None,None,25,30),get_gzsl_taskAL48_jk,"GZSL"),
    "GOSR": partial(dan_eval_cfg_common,modcfg(None,None,25,30), get_gosr_taskAL48_jpn,"GOSR"),
    "OSTR": partial(dan_eval_cfg_common,modcfg(None,None,25,30), get_ostr_taskAL48_jpn,"OSTR"),
}
