"""
 Created by Myung-Joon Kwon
 mjkwon2021@gmail.com
 June 7, 2021
"""
import os
import pdb
import sys

import cv2
from PIL import Image
from torchvision.transforms import transforms
from unet.UNet_3Plus import UNet_3Plus
from utils import Inference_single

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from lib.config import config
from lib.config import update_config

from torch.nn import functional as F
from data.data_core import SplicingDataset as splicing_dataset
from project_config import dataset_paths


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,default='./DDT.yaml')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--gt_file', type=str,default=r"./data/pred.txt")

    parser.add_argument('--model_name', type=str,default='loss-0.0164')

    args = parser.parse_args()
    update_config(config, args)

    return args


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])


def main():

    opt = parse_args()
    model_name = opt.model_name
    test_file = opt.gt_file
    testDataName = os.path.split(test_file)[-1].split('.')[0]


    os.makedirs(f'../save_out/{model_name}/{testDataName}', exist_ok=True)
    args = argparse.Namespace(cfg='./DDT.yaml', opts=['TEST.MODEL_FILE', f'./output/{model_name}.pkl',
                                                    'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])

    update_config(config, args)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    #!!!   None
    test_dataset = splicing_dataset(crop_size=[640,640], mode='arbitrary',file_path=test_file)
    print(test_dataset.get_info())

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # must be 1 to handle arbitrary input sizes
        shuffle=False,  # must be False to get accurate filename
        num_workers=1,
        pin_memory=False)

    model = UNet_3Plus(in_channels=3, n_classes=1)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        raise ValueError("Model file is not specified.")
    print('=> loading model from {}'.format(model_state_file))
    model = Inference_single(model)
    checkpoint = torch.load(model_state_file)
    model.load_state_dict(checkpoint['state_dict'])

    # dataset_paths['SAVE_PRED'].mkdir(parents=True, exist_ok=True)

    def get_next_filename(i):
        dataset_list = test_dataset.dataset_list
        it = 0
        while True:
            if i >= len(dataset_list[it]):
                i -= len(dataset_list[it])
                it += 1
                continue
            name = dataset_list[it].get_tamp_name(i)
            # print(f"{name}")
            name = os.path.split(name)[-1]
            return name

    def restore_to_original_size(cropped_img, original_size):
        h, w = original_size
        restored_img = cropped_img[:h, :w]
        return restored_img

    with (torch.no_grad()):
        num=0
        for index, (image, true_masks,original_size) in enumerate(tqdm(testloader)):
            name = os.path.splitext(get_next_filename(index))[0]

            # size = true_masks.size()
            # image = F.interpolate(image, size=(size, size), mode='bilinear',
            #                             align_corners=False)


            image = image.cuda()
            model.eval()

            pred = model(image)
            # pred = (np.array(pred) > 0.5)
            pred = [np.array(transform_pil(pred[i])) for i in range(len(pred))]
            if len(pred) != 1:
                pdb.set_trace()
            else:
                pred = pred[0]
            filename = name + ".png"
            save_seg_path = os.path.join(f'./save_out/{model_name}/{testDataName}', filename)
            os.makedirs(os.path.split(save_seg_path)[0], exist_ok=True)

            #!!!
            # pred=cv2.resize(pred,(original_size[1].item(),original_size[0].item()))
            pred = restore_to_original_size(pred, (original_size))
            cv2.imwrite(save_seg_path, pred.astype(np.uint8))


if __name__ == '__main__':
    main()
