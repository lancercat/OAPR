from torchvision import transforms

from neko_2020nocr.dan.configs.datasets.ds_paths import get_nips14, get_cvpr16, get_SVT, get_iiit5k, get_cute, \
    get_IC03_867, get_IC13_1015
from neko_2021_mjt.configs.data.chs_jpn_data import randomsampler
from neko_2022_soai_zero.aas_release.configs.data.chs_jpn_data_aligned_left import get_eval_word_color_core_AL
from neko_2022_soai_zero.aas_release.dataloaders.dataset_aligned_left import \
    colored_lmdbDataset_repeatHS_left_aligned


def get_mjstcqa_AL(root,maxT,bsize=48,rep=1,hw=[32,128]):
    if(rep>=0):
        dla={
                'batch_size': bsize,
                'shuffle': True,
                'num_workers': 8,
            }
    elif(rep==-1):
        dla = {
            'batch_size': bsize,
            "sampler": randomsampler(None),
            'num_workers': 8,
        }

    return \
    {

        "type": colored_lmdbDataset_repeatHS_left_aligned,
        'ds_args':
        {
            "repeat": rep,
            'roots': [get_nips14(root),
                      get_cvpr16(root)],
            'img_height': hw[0],
            'img_width': hw[1],
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT": maxT,
            "qhb_aug": True
        },
        'dl_args':dla,
    }

def get_eval_mjst_colorAL(root,maxT,hw=[32,128],batch_size=32):
    return {
        "dict_dir":None,
        "case_sensitive": False,
        "te_case_sensitive": False,
            "datasets":{
                "CUTE": get_eval_word_color_core_AL(get_cute(root),maxT, hw,batch_size),
                "IIIT5k": get_eval_word_color_core_AL(get_iiit5k(root),maxT, hw,batch_size),
                "SVT": get_eval_word_color_core_AL(get_SVT(root),maxT, hw,batch_size),
                "IC03": get_eval_word_color_core_AL(get_IC03_867(root),maxT, hw,batch_size),
                "IC13": get_eval_word_color_core_AL(get_IC13_1015(root),maxT, hw,batch_size),
            }
        }
