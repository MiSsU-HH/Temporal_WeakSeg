import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


cams_path = "/data/zhuyan/dataset/cams/0402_p2_8gpus_test_lcm_dilation2_edge3/lcm_1"
gt_path = "/data/zhuyan/dataset/PASTIS24/"
csv_file = "/data/zhuyan/dataset/PASTIS24/fold-paths/folds_1_123_paths.csv"
data_list = pd.read_csv(csv_file, header=None)
thr_top = 3.1
thr_bottom = 3.1
nums_eq = 0
nums = 0
out_label_path = "/data/zhuyan/dataset/vis/label"
cate_num = np.zeros(19)
cate_true = np.zeros(19)

for idx in tqdm(range(len(data_list)), total=len(data_list)):
    img_name = os.path.join(gt_path, data_list.iloc[idx, 0])
    with open(img_name, 'rb') as handle:
        sample = pickle.load(handle, encoding='latin1')
    label = sample["labels"]
    label[label==19] = 0
    cams = np.load(os.path.join(cams_path, os.path.splitext(os.path.basename(img_name))[0] + ".npy"))
    cams = zoom(cams, zoom=(1, 2, 2), order=1)
    max_indices = np.argmax(cams, axis=0) + 1
    max_values = np.amax(cams, axis=0)
    mask = np.ones_like(max_values) * 255
    mask[max_values<thr_bottom] = 0
    mask[max_values>=thr_top] = 1
    max_indices[mask==0] = 0
    max_indices[mask==255] = 255
    max_indices = max_indices.astype(np.uint8)
    
    
    cv2.imwrite(os.path.join(out_label_path, os.path.splitext(os.path.basename(img_name))[0] + ".png"), max_indices)
    
    nums_eq = nums_eq+np.sum(np.equal(label, max_indices))
    nums = nums + np.count_nonzero(max_indices != 255)
    for i in range(24):
        for j in range(24):
            if max_indices[i,j]!=255:
                cate_num[max_indices[i,j]] = cate_num[max_indices[i,j]]+1
                if max_indices[i,j]==label[i,j]:
                    cate_true[max_indices[i,j]] = cate_true[max_indices[i,j]]+1

print("bottom: "+str(thr_bottom) + "  top: "+ str(thr_top))
print("acc: " + str(nums_eq/nums))
print("efficient: " + str(nums/(len(data_list)*24*24)))
print(cate_num)
print(cate_true)
print(cate_true/cate_num)