import scipy.io
import numpy as np

def replace(path_to_mask:str, idx:str, replace_list:dict):
    mask = scipy.io.loadmat(path_to_mask)[idx]
    size = mask.shape
    res_mask = np.zeros(size)
    for label in replace_list:
        res_mask[mask == label] = int(replace_list[label])
    return res_mask.astype("uint8")
