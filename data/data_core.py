import torch
from torch.utils.data import Dataset
import random

import project_config
from data.dataset_Load import dataset_Load
from data.dataset_Load_test import dataset_Load_test



class SplicingDataset(Dataset):
    def __init__(self, crop_size, mode="train", file_path=None):
        self.dataset_list = []
        if mode == "train":
            self.dataset_list.append(
               dataset_Load(project_config.dataset_paths['CASIA2'], crop_size,
                            "data/xxx.txt"))
        elif mode == "valid":
            self.dataset_list.append(dataset_Load(project_config.dataset_paths['CASIA2'],crop_size,  "data/xxx.txt"))

        elif mode == "arbitrary":
            self.dataset_list.append(dataset_Load_test( crop_size,file_path))
        else:
            raise KeyError("Invalid mode: " + mode)
        self.crop_size = crop_size
        self.mode = mode

    def shuffle(self):
        for dataset in self.dataset_list:
            random.shuffle(dataset.tamp_list)

    def get_PIL_image(self, index):
        it = 0
        while True:
            if index >= len(self.dataset_list[it]):
                index -= len(self.dataset_list[it])
                it += 1
                continue
            return self.dataset_list[it].get_PIL_Image(index)

    def get_filename(self, index):
        it = 0
        while True:
            if index >= len(self.dataset_list[it]):
                index -= len(self.dataset_list[it])
                it += 1
                continue
            return self.dataset_list[it].get_tamp_name(index)

    def __len__(self):
        return sum([len(lst) for lst in self.dataset_list])


    def __getitem__(self, index):
        it = 0
        while True:
            if index >= len(self.dataset_list[it]):
                index -= len(self.dataset_list[it])
                it += 1
                continue
            return self.dataset_list[it].get_tamp(index)



    def get_info(self):
        s = ""
        for ds in self.dataset_list:
            s += (str(ds)+'('+str(len(ds))+') ')
        s += '\n'
        s += f"crop_size={self.crop_size},  mode={self.mode}\n"
        return s





