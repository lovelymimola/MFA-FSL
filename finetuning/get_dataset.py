import numpy as np
import random
from sklearn.model_selection import train_test_split
import torch

def load_pretrain_dataset(num):
    x = np.load(f"H:/Dataset/16WiFi/KRI-16Devices-RawData/62ft_normalized/X_train_{num}Class_run1.npy")
    y = np.load(f"H:/Dataset/16WiFi/KRI-16Devices-RawData/62ft_normalized/Y_train_{num}Class_run1.npy")
    y = y.astype(np.uint8)
    return x, y

def load_train_dataset_k_shot(num):
    x = np.load(f"H:/Dataset/16WiFi/KRI-16Devices-RawData/62ft_normalized/X_train_{num}Class_run1.npy")
    y = np.load(f"H:/Dataset/16WiFi/KRI-16Devices-RawData/62ft_normalized/Y_train_{num}Class_run1.npy")
    y = y.astype(np.uint8)
    return x, y

def load_test_dataset(num):
    x1 = np.load(f"H:/Dataset/16WiFi/KRI-16Devices-RawData/62ft_normalized/X_test_{num}Class_run1_-10dB.npy")
    y1 = np.load(f"H:/Dataset/16WiFi/KRI-16Devices-RawData/62ft_normalized/Y_test_{num}Class_run1_-10dB.npy")
    y1 = y1.astype(np.uint8)

    x2 = np.load(f"H:/Dataset/16WiFi/KRI-16Devices-RawData/62ft_normalized/X_test_{num}Class_run2_-10dB.npy")
    y2 = np.load(f"H:/Dataset/16WiFi/KRI-16Devices-RawData/62ft_normalized/Y_test_{num}Class_run2_-10dB.npy")
    y2 = y2.astype(np.uint8)

    return x1, y1, x2, y2