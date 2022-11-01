import numpy as np
import torch
import torch.nn.functional as F
import scipy.io
import os
import nibabel as nib
import pdb

# (91, 109, 91) --> (96, 112, 96)


for dir in os.listdir("./nifti_files"):
    path = "./nifti_files/" + dir
    for filedir in os.listdir(path):
        for file in os.listdir(path + "/" + filedir):

            img = nib.load(path + "/" + filedir + "/" + file)
            pdb.set_trace()
            img_resized = np.pad(img, ((0,0), (0,0), (0,0)), 'constant')
            nifti = nib.Nifti1Image(img_resized, affine=np.eye(4))

            filename = file[:-4] + "_resized.nii"
            #nib.save(nifti, path + "/" + filedir + "/" + filename)

