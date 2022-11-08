import nibabel as nib
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision

subject_id = '52'

target_path = "./examples/brats2020/nifti_files/Validate/"
pred_path = "./predictions/validation/2layers/"
input_path = "./examples/brats2020/nifti_files/Validate/"

target = nib.load(target_path + 'S_{}/S_{}_seg_resized.nii'.format(subject_id, subject_id)).get_fdata()
pred = nib.load(pred_path + 'S_{}.nii'.format(subject_id)).get_fdata()
input = nib.load(input_path + 'S_{}/S_{}_t1_resized.nii'.format(subject_id, subject_id)).get_fdata()

fig_rows = 4
fig_cols = 4
n_subplots = fig_rows * fig_cols
n_slice = input.shape[0]
step_size = n_slice // n_subplots
plot_range = n_subplots * step_size
start_stop = int((n_slice - plot_range) / 2)

fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])
fig2, axs2 = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])
# fig3, axs3 = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

#Fig1: input
for idx, img in enumerate(range(start_stop, plot_range, step_size)):
    axs.flat[idx].imshow(input[img, :, :], cmap='gray')
    axs.flat[idx].imshow(target[img, :, :], alpha=0.5)
    axs.flat[idx].axis('off')

for idx, img in enumerate(range(start_stop, plot_range, step_size)):
    axs2.flat[idx].imshow(input[img, :, :], cmap='gray')
    axs2.flat[idx].imshow(pred[img, :, :], alpha=0.5)
    axs2.flat[idx].axis('off')

# #Fig2: target
# for idx, img in enumerate(range(start_stop, plot_range, step_size)):
#     axs2.flat[idx].imshow(target[img, :, :], cmap='gray')
#     axs2.flat[idx].axis('off')
#
# #Fig3: prediction
# for idx, img in enumerate(range(start_stop, plot_range, step_size)):
#     axs3.flat[idx].imshow(pred[img, :, :], cmap='gray')
#     axs3.flat[idx].axis('off')

plt.tight_layout()
plt.show()