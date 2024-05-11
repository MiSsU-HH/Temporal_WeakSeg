# Temporal_WeakSeg
Weakly supervised semantic segmentation of temporal data.


conda env path: /data/zhuyan/anaconda3/envs/deepsatmodels


**step1: train cls model**

data/datasets.yaml

    PASTIS24_cls:

        paths_train: "/data/zhuyan/dataset/PASTIS24/fold-paths/folds_1_123_paths.csv"

        paths_eval: "/data/zhuyan/dataset/PASTIS24/fold-paths/fold_4_paths.csv"

configs/PASTIS24/TSViT_cls.yaml

    save_path: "PATH_TO_SAVE"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_and_eval/classification_train_transf.py --config configs/PASTIS24/TSViT_cls.yaml

**step2: generate cam**

data/datasets.yaml

    PASTIS24_cls:

        paths_train: "/data/zhuyan/dataset/PASTIS24/fold-paths/folds_1_123_paths.csv"

        paths_eval: "/data/zhuyan/dataset/PASTIS24/fold-paths/folds_1_123_paths.csv"

configs/PASTIS24/TSViT_cls.yaml

    load_from_checkpoint: "CKPT_OF_STEP1"

train_and_eval/classification_train_transf.py

    line145: feature_map_path="SAVE_PATH"

python train_and_eval/classification_train_transf.py --config configs/PASTIS24/TSViT_cls.yaml --device 0 --generate_cams

**step3: generate pseduo label**

todo

**step4: segmentaion**

todo