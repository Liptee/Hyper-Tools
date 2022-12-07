import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from scripts.utils import delete_zero
from scripts.utils import save_image_as

def report(path_to_Y:str, path_to_predict:str, idx:str = "img", zero_delete = False):
    Y = scipy.io.loadmat(path_to_Y)[idx]
    predict = cv2.imread(path_to_predict, cv2.IMREAD_GRAYSCALE)
    if zero_delete:
        Y = delete_zero(Y)
        predict = delete_zero(predict)
    plt.figure(figsize=(12,12))
    plt.imshow(Y)
    plt.figure(figsize=(12,12))
    plt.imshow(predict)
    Y = list(Y.flatten())
    predict = list(predict.flatten())
    print(classification_report(Y, predict))

def create_ansamle(path_to_Y:str, path_to_multi:str, path_to_dir_of_binary:str, list_of_labels:list, idx:str = "img"):
    multi = cv2.imread(path_to_multi, cv2.IMREAD_GRAYSCALE)
    copy = np.copy(multi)
    for i in list_of_labels:
        binary = cv2.imread(f"{path_to_dir_of_binary}\plus_mask_for_{i}_comp.mat.png", cv2.IMREAD_GRAYSCALE)
        copy[binary == 2] = i
    save_image_as("ansamble/an_predict.png", copy)
    report(path_to_Y, "ansamble/an_predict.png", zero_delete=True)