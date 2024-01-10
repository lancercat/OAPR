from torchvision import transforms

from neko_2020nocr.dan.methods_pami.pami_osds_paths import get_lsvtK_path, get_ctwK_path, \
    get_mlt_chlatK_path, get_artK_path, get_rctwK_path, get_mltjp_path, get_mltkr_path
from neko_2021_mjt.configs.data.chs_jpn_data import randomsampler
from neko_2022_soai_zero.aas_release.dataloaders.dataset_aligned_left import \
    colored_lmdbDataset_repeatHS_left_aligned, colored_lmdbDataset_left_aligned


def get_chs_HScqa_AL(root,maxT,bsize=48,rep=1,hw=[32,128]):
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
            'roots': [get_lsvtK_path(root),
                      get_ctwK_path(root),
                      get_mlt_chlatK_path(root),
                      get_artK_path(root),
                      get_rctwK_path(root)
                      ],
            'img_height': hw[0],
            'img_width': hw[1],
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Train',
            "maxT": maxT,
            "qhb_aug": True
        },
        'dl_args':dla,
    }


def get_eval_word_color_core_AL(teroot,maxT,hw=[32,128],batch_size=32):
    return \
    {
        "type": colored_lmdbDataset_left_aligned,
        'ds_args':
        {
            'roots': [teroot],
            'img_height': hw[0],
            'img_width': hw[1],
            'transform': transforms.Compose([transforms.ToTensor()]),
            'global_state': 'Test',
            "maxT": maxT
        },
        'dl_args':
        {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 24,
        },
    }
def get_eval_jpn_colorAL(root,maxT,hw=[32,128]):
    teroot= get_mltjp_path(root);
    return {
        "dict_dir":None,
        "case_sensitive": False,
        "te_case_sensitive": False,
            "datasets":{
                "JAP_lang": get_eval_word_color_core_AL(teroot,maxT,hw),
            }
        }

def get_eval_kr_colorAL(root,maxT,hw=[32,128]):
    teroot= get_mltkr_path(root);
    return {
        "dict_dir":None,
        "case_sensitive": False,
        "te_case_sensitive": False,
            "datasets":{
                "KR_lang": get_eval_word_color_core_AL(teroot,maxT,hw),
            }
        }