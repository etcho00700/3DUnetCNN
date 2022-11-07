import nibabel as nib
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision
from skimage.transform import rotate
from skimage.util import montage

brats_id = '012'

target_path = "./examples/brats2020/brats2020/MICCAI_BraTS2020_TrainingData/"
pred_path = "./predictions/validation/baseline/"
input_path = "./examples/brats2020/brats2020/MICCAI_BraTS2020_TrainingData/"

input_49 = nib.load(target_path + 'BraTS20_Training_009/BraTS20_Training_009_t1.nii').get_fdata()
pred_49 = nib.load(pred_path + 'Brats_prediction_009.nii.gz').get_fdata()
target_49 = nib.load(input_path + 'BraTS20_Training_009/BraTS20_Training_009_seg.nii').get_fdata()

fig_rows = 4
fig_cols = 4
n_subplots = fig_rows * fig_cols
n_slice = input_49.shape[0]
step_size = n_slice // n_subplots
plot_range = n_subplots * step_size
start_stop = int((n_slice - plot_range) / 2)

fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])
fig2, axs2 = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])
fig3, axs3 = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

#Input
for idx, img in enumerate(range(start_stop, plot_range, step_size)):
    axs.flat[idx].imshow(input_49[img, :, :], cmap='gray')
    axs.flat[idx].axis('off')

# #Target
# for idx, img in enumerate(range(start_stop, plot_range, step_size)):
#     axs2.flat[idx].imshow(target_49[img, :, :], cmap='gray')
#     axs2.flat[idx].axis('off')
#
# #Prediction
# for idx, img in enumerate(range(start_stop, plot_range, step_size)):
#     axs3.flat[idx].imshow(pred_49[img, :, :], cmap='gray')
#     axs3.flat[idx].axis('off')

plt.tight_layout()
plt.show()