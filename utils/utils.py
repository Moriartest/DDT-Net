import logging
import os
import pdb
import random
import time
from collections import Counter
from distutils import dist
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F

def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    # this function make img to square due to network architecture of UNet
    # 。如果pos为0，则返回img的前h列；如果pos为1，则返回img的后h列。
    # 这个函数主要用于将图像转换为正方形的形状，以适应UNet网络的架构。由于UNet要求输入图像为正方形，因此通过提取左侧或右侧的正方形部分来实现形状的转换
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]


def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255


def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def read_annotations(data_path):
    lines = map(str.strip, open(data_path).readlines())
    data = []
    for line in lines:
        temp = line.split(',')
        if len(temp) == 1:
            sample_path = temp[0]
            mask_path = 'None'
        else:
            sample_path, mask_path = temp
        data.append((sample_path, mask_path))
    return data


def calculate_img_score(pd, gt):
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
    acc = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos + 1e-6)
    sen = true_pos / (true_pos + false_neg + 1e-6)
    spe = true_neg / (true_neg + false_pos + 1e-6)
    f1 = 2 * sen * spe / (sen + spe)
    return acc, sen, spe, f1, true_pos, true_neg, false_pos, false_neg


def calculate_pixel_f1(pd, gt):
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        f1, iou = 1.0, 1.0
        return f1, 0.0, 0.0
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    return f1, precision, recall


def calculate_pixel(confu_mat_total, save_path=None):
    '''
    :param confu_mat: 总的混淆矩阵
    save_path：保存txt的路径
    :return: txt写出指标
    '''
    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float32)
    col_sum = np.sum(confu_mat, axis=1)  # 按行求和
    raw_sum = np.sum(confu_mat, axis=0)  # 每一列的数量

    pe_fz = 0
    PA = 0  # 像素准确率
    CPA = []  # 类别像素准确率
    TP = []  # 识别中每类分类正确的个数
    for i in range(class_num):
        pe_fz += col_sum[i] * raw_sum[i]
        PA = PA + confu_mat[i, i]
        CPA.append(confu_mat[i, i] / col_sum[i])
        TP.append(confu_mat[i, i])

    pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
    kappa = (PA - pe) / (1 - pe)  # Kappa系数
    PA = PA / confu_mat.sum()
    CPA = np.array(CPA)
    MPA = np.mean(CPA)  # 类别平均像素准确率

    # 计算f1-score
    TP = np.array(TP)
    FN = col_sum - TP
    FP = raw_sum - TP

    # 计算并写出f1_score,IOU,Mf1,MIOU
    f1_score = []  # 每个类别的f1_score
    IOU = []  # 每个类别的IOU
    for i in range(class_num):
        # 写出f1-score
        f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
        f1_score.append(f1)
        iou = TP[i] / (TP[i] + FP[i] + FN[i])
        IOU.append(iou)

    f1_score = np.array(f1_score)
    Mf1 = np.mean(f1_score)  # f1_score的平均值
    IOU = np.array(IOU)
    MIOU = np.mean(IOU)  # IOU的平均值

    return f1_score, Mf1, IOU, MIOU


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                          (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def get_confusion_matrix(label, output, size, num_class=1):
    """
    Calcute the confusion matrix by given label and pred
    """
    # output = pred.cpu().numpy()
    seg_pred = np.asarray(np.round(output), dtype=np.uint8)
    seg_gt = np.asarray(
        label[:, :size[-2], :size[-1]], dtype=np.int)

    index = (seg_gt * num_class + seg_pred).astype('int32')
    index = np.ravel(index)
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                i_pred] = label_count[cur_index]
    return confusion_matrix


# 计算混淆矩阵
def cal_confu_matrix(label, predict, class_num):
    confu_list = []
    for i in range(class_num):
        c = Counter(predict[np.where(label == i)])
        single_row = []
        for j in range(class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int32)


class FullModel(nn.Module):
    """
    Distribute the loss on multi-gpu to reduce
    the memory cost in the main gpu.
    You can check the following discussion.
    https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
    """

    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, inputs, true_masks):
        outputs = self.model(inputs)
        outputs = torch.sigmoid(outputs)
        outputs = torch.squeeze(outputs, 1)
        loss = self.loss(outputs, true_masks)
        return torch.unsqueeze(loss, 0), outputs

class FullModelForUnet3DeepSup(nn.Module):
    """
    Distribute the loss on multi-gpu to reduce
    the memory cost in the main gpu.
    You can check the following discussion.
    https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
    """

    def __init__(self, model, loss):
        super(FullModelForUnet3DeepSup, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, inputs, true_masks):
        d1,d2,d3,d4,d5 = self.model(inputs)
        #!!!
        # outputs = torch.sigmoid(outputs)
        d1 = torch.squeeze(d1, 1)
        d2 = torch.squeeze(d2, 1)
        d3 = torch.squeeze(d3, 1)
        d4 = torch.squeeze(d4, 1)
        d5 = torch.squeeze(d5, 1)


        loss1 = self.loss(d1, true_masks.float())
        loss2 = self.loss(d2, true_masks.float())
        loss3 = self.loss(d3, true_masks.float())
        loss4 = self.loss(d4, true_masks.float())
        loss5  = self.loss(d5, true_masks.float())
        loss=loss1+loss2+loss3+loss4+loss5
        return torch.unsqueeze(loss, 0), d1

windows_mode = True
def get_world_size():
    if windows_mode:
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp

transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])


class Inference_single(nn.Module):

    def __init__(self, model):
        super(Inference_single, self).__init__()
        self.model = model
        # self.loss = loss

    def forward(self, inputs):
        outputs = self.model(inputs)
        out = torch.sigmoid(outputs).detach().cpu()

        # outputs_sig=torch.squeeze(out,1)

        # seg = [np.array(transform_pil(out[i])) for i in range(len(out))]
        # out = seg[0]
        return out
def img_to_tensor(im, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(im / (255.0 if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor

def direct_val(imgs):
    normalize = {"mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225]}
    if len(imgs) != 1:
        pdb.set_trace()
    imgs = img_to_tensor(imgs[0], normalize).unsqueeze(0)
    return imgs

class inference_single_2(nn.Module):

    def __init__(self, model):
        super(inference_single_2, self).__init__()
        self.model = model

    def forward(self, img,th=0):
        self.model.eval()
        with torch.no_grad():
            img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))
            img = direct_val(img)
            img = img.cuda()
            seg = self.model(img)
            seg = torch.sigmoid(seg).detach().cpu()
            if torch.isnan(seg).any() or torch.isinf(seg).any():
                max_score = 0.0
            else:
                max_score = torch.max(seg).numpy()
            seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]

            if len(seg) != 1:
                pdb.set_trace()
            else:
                fake_seg = seg[0]
            if th == 0:
                return fake_seg, max_score
            fake_seg = 255.0 * (fake_seg > 255 * th)
            fake_seg = fake_seg.astype(np.uint8)

        return fake_seg

def adjust_learning_rate(optimizer, base_lr, max_iters,
                         cur_iters, power=0.9):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    return lr


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        # ph, pw = score.size(1), score.size(2)
        # h, w = target.size(0), target.size(1)
        # if ph != h or pw != w:
        #     score = F.upsample(
        #         input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss


def Metrics(predict_image, gt_image):
    # 将图像转换为二进制数组
    predicted_mask = np.array((predict_image > 0.5), dtype=bool)
    true_mask = np.squeeze(np.array(gt_image, dtype=bool))

    # # 计算True Positive（TP）
    # tp = np.sum(np.logical_and(predict_image, gt_image))
    #
    # # 计算True Negative（TN）
    # tn = np.sum(np.logical_and(np.logical_not(predict_image), np.logical_not(gt_image)))
    #
    # # 计算False Positive（FP）
    # fp = np.sum(np.logical_and(predict_image, np.logical_not(gt_image)))
    #
    # # 计算False Negative（FN）
    # fn = np.sum(np.logical_and(np.logical_not(predict_image), gt_image))

    # 真正例（True Positive, TP）: 预测和真实掩码都为真
    tp = np.sum((predicted_mask == 1) & (true_mask == 1))

    # 假正例（False Positive, FP）: 预测为真，但真实掩码为假
    fp = np.sum((predicted_mask == 1) & (true_mask == 0))

    # 真负例（True Negative, TN）: 预测和真实掩码都为假
    tn = np.sum((predicted_mask == 0) & (true_mask == 0))

    # 假负例（False Negative, FN）: 预测为假，但真实掩码为真
    fn = np.sum((predicted_mask == 0) & (true_mask == 1))

    # 计算IOU（Intersection over Union）
    if (tp + fn + fp == 0):
        iou = 0
    else:
        iou = tp / (tp + fn + fp)

    # 计算Dice Coefficient（Dice系数）
    dice_coefficient = 2 * tp / (2 * tp + fn + fp)

    # 计算Accuracy（准确率）
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    # 计算precision（精确率）
    if (tp + fp == 0):
        precision = 0
    else:
        precision = tp / (tp + fp)

    # 计算recall（召回率）
    recall = tp / (tp + fn)

    # 计算Sensitivity（敏感度）
    sensitivity = tp / (tp + fn)

    # 计算F1-score
    if (precision + recall == 0):
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    # 计算Specificity（特异度）
    specificity = tn / (tn + fp)

    return iou, dice_coefficient, accuracy, precision, recall, sensitivity, f1, specificity


def rgb2gray(rgb):
    b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = torch.unsqueeze(gray, 1)
    return gray