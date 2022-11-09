from torchmetrics import ConfusionMatrix
import torch
import nibabel as nib
import numpy as np
import seg_metrics.seg_metrics as seg
import pandas as pd
import pdb
import math

def DICE_COE(mask1, mask2):
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3) # for easy reading
    return dice


target_path = "./examples/brats2020/nifti_files/Validate/"
predict_path = "./predictions/validation/1layer/"
input_path = "./examples/brats2020/nifti_files/Validate/"

'''
Model1:
S_53: 0.594
S_50: 0.084

Model2:
S_49 DICE: 0.795
S_52 DICE: 0.924
S_50 DICE: 0.404

Model3:
S_49 DICE: 0.754
S_52 DICE: 0.927
S_50 DICE: 0.576
'''
labels = [0,1]
csv_file = "metrics1.csv"
for i in range(49, 55):
     subject_id = str(i)
     target = nib.load(target_path + 'S_{}/S_{}_seg_resized.nii'.format(subject_id, subject_id)).get_fdata()
     pred = nib.load(predict_path + 'S_{}.nii'.format(subject_id)).get_fdata()
     input = nib.load(input_path + 'S_{}/S_{}_t1_resized.nii'.format(subject_id, subject_id)).get_fdata()

     gdth_path = target_path + 'S_{}/S_{}_seg_resized.nii'.format(subject_id, subject_id)
     pred_path = predict_path + 'S_{}.nii'.format(subject_id)
     metrics = seg.write_metrics(labels=labels[1:], gdth_path=gdth_path, pred_path=pred_path,
                                 csv_file = csv_file)
     print(metrics)

data = pd.read_csv("./metrics1.csv")
mean_dice = data['dice'].mean()
mean_jaccard = data['jaccard'].mean()
mean_precision = data['precision'].mean()
mean_recall = data['recall'].mean()
mean_hd95 = data['hd95'].mean()

print(mean_dice)
print(mean_jaccard)
print(mean_precision)
print(mean_recall)
print(mean_hd95)