import scipy.io

def cutter(
    path_to_img:str,
    idx_in_img:str,
    path_to_mask:str,
    idx_in_mask:str,
    x:list,
    y:list,
    savename:str):
    #cut imgs and mask in .mat file. Save as savename.
    img = scipy.io.loadmat(path_to_img)[idx_in_img]
    mask = scipy.io.loadmat(path_to_mask)[idx_in_mask]
    x1 = x[0]
    x2 = x[1]
    y1 = y[0]
    y2 = y[1]
    mask = mask[x1:x2,y1:y2]
    img = img[x1:x2,y1:y2,:]
    mat_img = scipy.io.loadmat(path_to_img)
    mat_img[idx_in_img] = img
    mat_mask = scipy.io.loadmat(path_to_mask)
    mat_mask[idx_in_mask] = mask

    scipy.io.savemat(f"cutter_res/{savename}.mat", mat_img, "cutter_res")
    scipy.io.savemat(f"cutter_res/{savename}_mask.mat", mat_mask, "cutter_res")