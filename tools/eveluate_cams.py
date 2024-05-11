import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from scipy.ndimage import zoom


cams_path = "/data/zhuyan/dataset/cams/0402_p2_8gpus_test/feature_map_refine_9000"
gt_path = "/data/zhuyan/dataset/PASTIS24/"
csv_file = "/data/zhuyan/dataset/PASTIS24/fold-paths/folds_1_123_paths.csv"
data_list = pd.read_csv(csv_file, header=None)

for thr in [1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4]:
    nums = 0

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
        max_values[max_values<thr] = 0
        max_values[max_values>=thr] = 1
        max_indices = max_indices * max_values
        max_indices = max_indices.astype(np.uint8)
        nums = nums+np.sum(np.equal(label, max_indices))
    print("thr = " + str(thr))
    print(nums/(len(data_list)*24*24))