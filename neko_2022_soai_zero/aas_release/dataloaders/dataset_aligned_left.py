from neko_2020nocr.dan.dataloaders.dataset_scene import lmdbDataset,colored_lmdbDataset_repeatHS
from neko_sdk.ocr_modules.io.data_tiding import neko_aligned_left_padding;

class colored_lmdbDataset_repeatHS_left_aligned(colored_lmdbDataset_repeatHS):
    def keepratio_resize(self, img):
        img, bmask = neko_aligned_left_padding(img, None,
                                          img_width=self.img_width,
                                          img_height=self.img_height,
                                          target_ratio=self.target_ratio,
                                          qhb_aug=self.qhb_aug,gray=False)
        return img,bmask

class colored_lmdbDataset_left_aligned(lmdbDataset):
    def keepratio_resize(self, img):
        img,bmask=neko_aligned_left_padding(img,None,
                                       img_width=self.img_width,
                                       img_height=self.img_height,
                                       target_ratio=self.target_ratio,
                                       qhb_aug=self.qhb_aug,gray=False)
        return img,bmask;
