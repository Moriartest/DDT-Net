from abc import ABC, abstractmethod
from PIL import Image, JpegImagePlugin
import numpy as np
import torch
import random
import cv2


class AbstractDataset(ABC):

    def __init__(self, crop_size):
        self._crop_size = crop_size

        self.tamp_list = None



    def _create_tensor(self, im_path, mask):

        img_RGB = np.array(Image.open(im_path).convert("RGB"))

        h, w = img_RGB.shape[0], img_RGB.shape[1]

        if mask is None:
            mask = np.zeros((h, w))

        crop_size = self._crop_size

        if crop_size is not None:
            if h < crop_size[0] or w < crop_size[1]:

                # # pad img_RGB
                temp = np.full((max(h, crop_size[0]), max(w, crop_size[1]), 3), 0)
                temp[:img_RGB.shape[0], :img_RGB.shape[1], :] = img_RGB
                img_RGB = temp
                #
                # # pad mask
                temp = np.full((max(h, crop_size[0]), max(w, crop_size[1])), 0)
                temp[:mask.shape[0], :mask.shape[1]] = mask
                mask = temp

            s_r = random.randint(0, max(h - crop_size[0], 0))
            s_c = random.randint(0, max(w - crop_size[1], 0))
            img_RGB = img_RGB[s_r:s_r + crop_size[0], s_c:s_c + crop_size[1], :]
            mask = mask[s_r:s_r + crop_size[0], s_c:s_c + crop_size[1]]

        t_RGB = torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 255.0


        return t_RGB, torch.tensor(mask, dtype=torch.float),(h,w)


    @abstractmethod
    def get_tamp(self, index):
        pass

    def get_tamp_name(self, index):
        item = self.tamp_list[index]
        if isinstance(item, list):
            return item[0]
        else:
            return item

    def get_PIL_Image(self, index):
        file = self.tamp_list[index][0]
        im = Image.fromarray(cv2.cvtColor(cv2.imread(str(self._root_path / file)), cv2.COLOR_BGR2RGB))
        return im

    def __len__(self):
        return len(self.tamp_list)
