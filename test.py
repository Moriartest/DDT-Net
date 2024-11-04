import os
import cv2
import numpy as np
import sys
import argparse
import pdb
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import pickle



def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--gt_file', type=str, default="./data/pred.txt")
    parser.add_argument('--th', type=float, default=0.5)
    parser.add_argument("--model_name", type=str, help="Path to the pretrained model",
                        default=f"loss-0.0164")
    args = parser.parse_args()
    return args

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

if __name__ == '__main__':
    opt = parse_args()
    annotation_file = opt.gt_file
    testDataName = os.path.split(annotation_file)[-1].split('.')[0]
    model_name = opt.model_name
    pred_dir="save_out/"+model_name+'/'+testDataName

    if not os.path.exists(annotation_file):
        print("%s not exists, quit" % annotation_file)
        sys.exit()
    annotation = read_annotations(annotation_file)
    scores, labs = [], []
    f1s = []
    hist, acc_list, cacc_list, macc_list, ioU_list, mIoU_list, fwIoU_list = [], [], [], [], [], [], []
    results = []
    for ix, (img, mask) in enumerate(tqdm(annotation)):
        pred_path = os.path.join(pred_dir,   os.path.basename(img).split('.')[0] + '.png')
        try:
            pred = cv2.imread(pred_path, 0) / 255.0
        except:
            print("%s not exists" % pred_path)
            continue
        f1 = 0
        if mask != 'None':
            labs.append(1)
            try:
                gt = cv2.imread( mask, 0) / 255.0
            except:
                pdb.set_trace()
            if pred.shape != gt.shape:
                print("%s size not match" % pred_path)
                continue
            pred = (pred > opt.th).astype(float)
            gt = (gt > opt.th).astype(float)
            try:
                f1 = metrics.f1_score(pred.flatten(), gt.flatten())
                #f1, p, r = calculate_pixel_f1(pred.flatten(), gt.flatten())
            except Exception as e:
                print("发生异常：", e)
                import pdb
                pdb.set_trace()
            f1s.append(f1)
        else:
            labs.append(0)

        score = np.max(pred)
        scores.append(score)

    fpr, tpr, thresholds = metrics.roc_curve((np.array(labs) > 0).astype(int), scores, pos_label=1)
    try:
        img_auc = metrics.roc_auc_score((np.array(labs) > 0).astype(int), scores)
    except:
        print("only one class")
        img_auc = 0.0
    with open(os.path.join(pred_dir,   'roc.pkl'), 'wb') as f:
        pickle.dump({'fpr': fpr, 'tpr': tpr, 'th': thresholds, 'auc': img_auc}, f)
        print("roc save at %s" % (os.path.join(pred_dir, 'roc.pkl')))


    meanf1 = np.mean(f1s)
    print("pixel-f1: %.4f" % meanf1)

    pred_labels = (np.array(scores) > 0.5).astype(int)
    y_true=(np.array(labs) > 0).astype(np.int)
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, pred_labels).ravel()

    # 计算特异度和灵敏度
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    accuracy = metrics.accuracy_score(labs, pred_labels)
    recall = metrics.recall_score(labs, pred_labels)
    precision = metrics.precision_score(labs, pred_labels)
    f1_imglevel = metrics.f1_score(labs, pred_labels)
    print("img level ：")
    print(f"img_f1: {f1_imglevel:.4f}     auc: {img_auc:.4f}   Sensitivity: {sensitivity:.4f}  Specificity:{specificity:.4f}  accuracy: {accuracy:.4f}  recall: {recall:.4f}    precision_score: {precision:.4f}    \n")


    with open(f'{pred_dir}/0test.txt', 'w') as f:
        f.write(f"pixel-f1: {meanf1:.4f}\n")
        f.write(f"img_f1: {f1_imglevel:.4f}     auc: {img_auc:.4f}   Sensitivity: {sensitivity:.4f}  Specificity:{specificity:.4f}  accuracy: {accuracy:.4f}  recall: {recall:.4f}    precision_score: {precision:.4f}    \n")
