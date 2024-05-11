import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool


def process_file(img_name):
    with open(img_name, 'rb') as handle:
        sample = pickle.load(handle, encoding='latin1')
    label = sample["labels"]
    label[label==19] = 0
    cams = np.load(os.path.join(cams_path, os.path.splitext(os.path.basename(img_name))[0] + ".npy"))
    max_indices = np.argmax(cams, axis=0) + 1
    max_values = np.amax(cams, axis=0)
    mask = np.ones_like(max_values) * 255
    mask[max_values<thr_bottom] = 0
    mask[max_values>=thr_top] = 1
    max_indices[mask==0] = 0
    max_indices[mask==255] = 255
    max_indices = np.repeat(max_indices, 2, axis=0)
    max_indices = np.repeat(max_indices, 2, axis=1)
    max_indices = max_indices.astype(np.uint8)

    """
    三合一拼图
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    rgb_pred = np.zeros((24, 24, 3), dtype=np.uint8)
    for i in range(24):
        for j in range(24):
            if max_indices[i, j]==255:
                rgb_pred[i, j] = [0, 0, 0]
            else:
                rgb_pred[i, j] = pat[max_indices[i, j]]

    rgb_label = np.zeros((24, 24, 3), dtype=np.uint8)
    for i in range(24):
        for j in range(24):
            rgb_label[i, j] = pat[label[i, j]]

    wrong_pred = np.ones((24, 24), dtype=np.uint8)*255
    for i in range(24):
        for j in range(24):
            if max_indices[i, j]==255:
                wrong_pred[i,j] = 0
            elif max_indices[i,j]==label[i,j]:
                wrong_pred[i,j] = 0

    axs[0].imshow(rgb_pred)
    axs[0].set_title('Original')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(wrong_pred, cmap='gray')
    axs[1].set_title('Dilated')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    axs[2].imshow(rgb_label)
    axs[2].set_title('Eroded')
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(vis_output_path, os.path.splitext(os.path.basename(img_name))[0] + ".png"))
    plt.close()


cams_path = "/data/zhuyan/dataset/cams/0402_p2_8gpus_test/9000"
gt_path = "/data/zhuyan/dataset/PASTIS24/"
csv_file = "/data/zhuyan/dataset/PASTIS24/fold-paths/folds_1_123_paths.csv"
vis_output_path = "/data/zhuyan/dataset/vis/0402_p2_8gpus_vis_5_1"
data_list = pd.read_csv(csv_file, header=None)
img_name_list = []

for idx in tqdm(range(len(data_list)), total=len(data_list)):
    img_name = os.path.join(gt_path, data_list.iloc[idx, 0])
    img_name_list.append(img_name)

thr_top = 3
thr_bottom = 3

pat = [
    [255, 255, 255],  # 类别0 (白色)
    [255, 0, 0],      # 类别1 (红色)
    [0, 255, 0],      # 类别2 (绿色)
    [0, 0, 255],      # 类别3 (蓝色)
    [255, 255, 0],    # 类别4 (黄色)
    [255, 0, 255],    # 类别5 (品红)
    [0, 255, 255],    # 类别6 (青色)
    [128, 0, 0],      # 类别7 (栗色)
    [0, 128, 0],      # 类别8 (深绿)
    [0, 0, 128],      # 类别9 (深蓝)
    [128, 128, 0],    # 类别10 (橄榄色)
    [128, 0, 128],    # 类别11 (紫色)
    [0, 128, 128],    # 类别12 (深青)
    [192, 192, 192],  # 类别13 (银色)
    [128, 128, 128],  # 类别14
    [255, 165, 0],    # 类别15
    [210, 105, 30],   # 类别16
    [255, 192, 203],  # 类别17
    [139, 69, 19]     # 类别18
]

with Pool() as pool:
    # 使用map方法并行执行任务
    pool.map(process_file, tqdm(img_name_list))


