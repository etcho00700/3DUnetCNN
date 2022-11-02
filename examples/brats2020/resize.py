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

            #if file[:-11] == "resized.nii":
                #os.remove(path + "/" + filedir + "/" + file)

            img = nib.load(path + "/" + filedir + "/" + file)
            img = np.array(img.dataobj)
            img_resized = np.pad(img, ((2,3), (2,1), (2,3)), 'constant')
            print(np.shape(img_resized))
            nifti = nib.Nifti1Image(img_resized, affine=np.eye(4))

            filename = file[:-4] + "_resized.nii"
            nib.save(nifti, path + "/" + filedir + "/" + filename)

#img = nib.load("./nifti_files/Train/S_1/S_1_seg_resized.nii")
#img = np.array(img.dataobj)
#print(np.shape(img))

#arr = np.random.rand(91,109,91)
#print(type(arr))
#resized = np.pad(arr, ((1,0),(0,0),(0,0)), 'constant')
