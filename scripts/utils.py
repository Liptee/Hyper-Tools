import os
import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def load_data(path:str, end:str):
    return sorted(glob(os.path.join(path, f"*{end}")))

def clear_dir(path, end):
    X = load_data(path, end)
    for x in X:
        os.remove(x)

def look_to(path_to_mask:str, idx:str):
    #only for .mat file
    mask = scipy.io.loadmat(path_to_mask)[idx]
    print(f"labels: {np.unique(mask)}")
    plt.imshow(mask)

def save_mat_as(savename:str, mask:np.ndarray, idx:str, path_to_orig:str, dir:str):
    orig = scipy.io.loadmat(path_to_orig)
    orig["img"] = mask
    scipy.io.savemat(f"{dir}/{savename}", orig)

def delete_zero(Y):
    copy = np.copy(Y)
    copy[Y == 0] = 1
    return copy

def save_image_as(savepath:str, img:np.ndarray):
    cv2.imwrite(savepath, img)

def hyper2RGB(path_to_mat:str, idx:str):
    img = scipy.io.loadmat(path_to_mat)[idx]
    deep = img.shape[2]
    print(deep)