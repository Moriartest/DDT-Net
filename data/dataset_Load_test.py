import project_config
from data.AbstractDataset import AbstractDataset

import os
import numpy as np
import random
from PIL import Image
from pathlib import Path
import torch


class dataset_Load_test(AbstractDataset):
    """
    directory structure
    tampCOCO (dataset_path["tampCOCO"] in project_config.py)
    ├── cm_images
    ├── cm_masks
    └── sp_images ...
    """

    def __init__(self,  crop_size, tamp_list: str):

        super().__init__(crop_size)
        with open(project_config.project_root / tamp_list, "r") as f:
            self.tamp_list = []
            for line in f.readlines():
                parts = line.strip().split(',')
                # if len(parts) < 2:
                #     print(f"Error: expected 2 parts but got {len(parts)} in line '{line.strip()}',tamp_list:{tamp_list}:")
                #     print(parts[0])
                # else:
                self.tamp_list.append(parts)

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path =  self.tamp_list[index][0]
        mask = None
        if (self.tamp_list[index][1] != "None"):
            mask_path = self.tamp_list[index][1]
            mask = np.array(Image.open(mask_path).convert('L'))
            mask[mask > 0] = 1
        return self._create_tensor(tamp_path, mask)
