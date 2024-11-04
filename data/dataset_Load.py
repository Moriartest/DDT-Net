import project_config
from data.AbstractDataset import AbstractDataset

import os
import numpy as np
import random
from PIL import Image
from pathlib import Path
import torch


class dataset_Load(AbstractDataset):


    def __init__(self, root_path, crop_size, tamp_list: str):

        super().__init__(crop_size)
        self._root_path = root_path
        with open(project_config.project_root / tamp_list, "r") as f:
            self.tamp_list = []
            for line in f.readlines():
                parts = line.strip().split(',')
                self.tamp_list.append(parts)

    def get_tamp(self, index):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path = self._root_path / self.tamp_list[index][0]
        mask = None
        if (self.tamp_list[index][1] != "None"):
            mask_path = self._root_path / self.tamp_list[index][1]
            mask = np.array(Image.open(mask_path).convert('L'))
            mask[mask > 0] = 1
        return self._create_tensor(tamp_path, mask)
