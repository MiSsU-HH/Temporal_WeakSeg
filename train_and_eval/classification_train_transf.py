import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
import torch.nn.functional as F
from metrics.mean_ap import mAP
from tqdm import tqdm


def mIou(y_true, y_pred, n_classes):
    """
    Mean Intersect over Union metric.
    Computes the one versus all IoU for each class and returns the average.
    Classes that do not appear in the provided set are not counted in the average.
    Args:
        y_true (1D-array): True labels
        y_pred (1D-array): Predicted labels
        n_classes (int): Total number of classes
    Returns:
        mean Iou (float)
    """
    iou = 0
    n_observed = n_classes
    for i in range(n_classes):
        y_t = (np.array(y_true) == i).astype(int)
        y_p = (np.array(y_pred) == i).astype(int)

        inter = np.sum(y_t * y_p)
        union = np.sum((y_t + y_p > 0).astype(int))

        if union == 0:
            n_observed -= 1
        else:
            iou += inter / union

    return iou / n_observed

def train_and_evaluate(net, dataloaders, config, device, lin_cls=False, generate_cams=False):

    def train_step(net, sample, loss_fn, optimizer, device, config):
        num_classes = config['MODEL']['num_classes']
        optimizer.zero_grad()
        semantic_label = sample['labels'].to(torch.int64).to(device)
        semantic_label[semantic_label==num_classes+1] = 0
        ground_truth = torch.zeros((semantic_label.shape[0], num_classes)).to(torch.int64).to(device)
        for i in range(semantic_label.shape[0]):
            unique_values, unique_nums = torch.unique(semantic_label[i,:,:], return_counts=True)
            for j in range(unique_values.shape[0]):
                unique_value = unique_values[j]
                unique_num = unique_nums[j]
                if unique_value!=0 and unique_num>semantic_label.shape[1]*semantic_label.shape[2]*0.01:
                    ground_truth[i,unique_value-1] = 1
        x_cls_logits, x_patch_logits = net(sample['inputs'].to(torch.float32).to(device))
        # x_patch_logits = net(sample['inputs'].to(torch.float32).to(device))
        
        loss_cls = F.multilabel_soft_margin_loss(x_cls_logits, ground_truth)
        loss_patch = F.multilabel_soft_margin_loss(x_patch_logits, ground_truth)
        loss = loss_cls + loss_patch
        # loss = loss_patch * 2
        loss.backward()
        optimizer.step()
        # return 0, loss_patch, loss
        return loss_cls, loss_patch, loss

    def evaluate(net, evalloader, loss_fn, config):
        num_classes = config['MODEL']['num_classes']
        Neval = len(evalloader)
        predicted_all = []
        labels_all = []
        net.eval()
        with torch.no_grad():
            for step, sample in tqdm(enumerate(evalloader), total=len(evalloader)):

                logits, x_patch_logits = net(sample['inputs'].to(torch.float32).to(device))
                # x_patch_logits = net(sample['inputs'].to(torch.float32).to(device))
                predicted = torch.sigmoid(logits).cpu().numpy()
                # predicted = torch.sigmoid(x_patch_logits).cpu().numpy()
                predicted_all.append(predicted)
                semantic_label = sample['labels'].to(torch.int64).cpu().numpy()
                semantic_label[semantic_label==num_classes+1] = 0
                ground_truth = np.zeros((semantic_label.shape[0], num_classes))
                for i in range(semantic_label.shape[0]):
                    unique_values, unique_nums = np.unique(semantic_label[i,:,:], return_counts=True)
                    for j in range(unique_values.shape[0]):
                        unique_value = unique_values[j]
                        unique_num = unique_nums[j]
                        if unique_value!=0 and unique_num>semantic_label.shape[1]*semantic_label.shape[2]*0.01:
                            ground_truth[i,unique_value-1] = 1
                labels_all.append(ground_truth)

        print("finished iterating over dataset after step %d" % step)
        print("calculating metrics...")
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)

        eval_results = {}
        mAP_value_macro, mAP_value_micro, ap = mAP(predicted_classes, target_classes)
        eval_results['mAP_macro'] = mAP_value_macro
        eval_results['mAP_micro'] = mAP_value_micro
        for i in range(np.size(ap)):
            eval_results[f"AP_class{i}"] = ap[i]
        print("-------------------------------------------------------------------------------------------------------")
        for key, value in eval_results.items():
            print(key + ': ' + str(value), end="   ")
        print("-------------------------------------------------------------------------------------------------------")

        return eval_results
    
    def generate_attention_maps_ms(net, evalloader, config):
        num_classes = config['MODEL']['num_classes']
        net.eval()
        with torch.no_grad():
            for step, sample in tqdm(enumerate(evalloader), total=len(evalloader)):
                semantic_label = sample['labels'].to(torch.int64).numpy()
                semantic_label[semantic_label==num_classes+1] = 0
                ground_truth = np.zeros((semantic_label.shape[0], num_classes), dtype=np.uint8)
                for i in range(semantic_label.shape[0]):
                    unique_values, unique_nums = np.unique(semantic_label[i,:,:], return_counts=True)
                    for j in range(unique_values.shape[0]):
                        unique_value = unique_values[j]
                        unique_num = unique_nums[j]
                        if unique_value!=0 and unique_num>semantic_label.shape[1]*semantic_label.shape[2]*0.01:
                            ground_truth[i,unique_value-1] = 1
                logits, x_patch_logits, cls_attn, feature_map, cams, cams_refine = net(sample['inputs'].to(torch.float32).to(device))
                # cls_attn = cls_attn.cpu().numpy() * ground_truth.reshape(ground_truth.shape[0], ground_truth.shape[1], 1, 1)
                feature_map = feature_map.cpu().numpy() * ground_truth.reshape(ground_truth.shape[0], ground_truth.shape[1], 1, 1)
                # cams = cams.cpu().numpy() * ground_truth.reshape(ground_truth.shape[0], ground_truth.shape[1], 1, 1)
                # cams_refine = cams_refine.cpu().numpy() * ground_truth.reshape(ground_truth.shape[0], ground_truth.shape[1], 1, 1)
                # cls_attn_path = "/data/zhuyan/dataset/cams/0325_-1_-1/cls_attn"
                feature_map_path = "/data/zhuyan/dataset/cams/0402_p2_8gpus_test/feature_map_refine_9000"
                if not os.path.exists(feature_map_path):
                    os.mkdir(feature_map_path)
                # cams_path = "/data/zhuyan/dataset/cams/0325_-1_-1/cams"
                # cams_refine_path = "/data/zhuyan/dataset/cams/0325_-1_-1/cams_refine"
                for i, img_path in enumerate(sample["img_path"]):
                    img_name = os.path.basename(img_path)
                    img_name, _ = os.path.splitext(img_name)
                #     cls_attn_output_path = os.path.join(cls_attn_path, img_name+".npy")
                    feature_map_output_path = os.path.join(feature_map_path, img_name+".npy")
                #     cams_output_path = os.path.join(cams_path, img_name+".npy")
                #     cams_refine_output_path = os.path.join(cams_refine_path, img_name+".npy")
                #     np.save(cls_attn_output_path,cls_attn[i,:,:,:])
                    np.save(feature_map_output_path,feature_map[i,:,:,:])
                #     np.save(cams_output_path,cams[i,:,:,:])
                #     np.save(cams_refine_output_path,cams_refine[i,:,:,:])



    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)

    start_global = 1
    start_epoch = 1
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False)
    print("current learn rate: ", lr)

    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    if generate_cams:
        generate_attention_maps_ms(net, dataloaders['eval'], config)
        return

    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    copy_yaml(config)

    loss_fn = {'all': get_loss(config, device, reduction=None),
               'mean': get_loss(config, device, reduction="mean")}

    if lin_cls:
        print('Train linear classifier only')
        trainable_params = get_net_trainable_params(net)[-2:]
    else:
        print('Train network end-to-end')
        trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    optimizer.zero_grad()

    scheduler = build_scheduler(config, optimizer, num_steps_train)

    writer = SummaryWriter(save_path)

    BEST_AP = 0
    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):  # loop over the dataset multiple times
        for step, sample in enumerate(dataloaders['train']):

            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step

            loss_cls, loss_patch, loss = train_step(net, sample, loss_fn, optimizer, device, config)

            if abs_step % train_metrics_steps == 0:
                batch_metrics = {"abs_step":abs_step,"epoch":epoch,"step":step + 1,"loss":loss,"loss_cls":loss_cls,"loss_patch":loss_patch,"lr":optimizer.param_groups[0]["lr"]}
                write_mean_summaries(writer, batch_metrics, abs_step, mode="train", optimizer=optimizer)
                print("abs_step: %d, epoch: %d, step: %d, loss: %.7f, loss_cls: %.7f, loss_patch: %.7f, lr: %.6f" %
                      (abs_step, epoch, step + 1, loss, loss_cls, loss_patch, optimizer.param_groups[0]["lr"]))
            if abs_step % eval_steps == 0:  # evaluate model every eval_steps batches
                eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, config)
                # if eval_metrics["mAP_macro"] > BEST_AP:
                #     if len(local_device_ids) > 1:
                #         torch.save(net.module.state_dict(), "%s/best.pth" % (save_path))
                #     else:
                #         torch.save(net.state_dict(), "%s/best.pth" % (save_path))
                #     BEST_AP = eval_metrics["mAP_macro"]
                if len(local_device_ids) > 1:
                    torch.save(net.module.state_dict(), "%s/%s.pth" % (save_path,str(abs_step)))
                else:
                    torch.save(net.state_dict(), "%s/%s.pth" % (save_path,str(abs_step)))
                write_mean_summaries(writer, eval_metrics, abs_step, mode="eval", optimizer=None)
                net.train()

        scheduler.step_update(abs_step)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', nargs='+', default=[0,1,2,3,4,5,6,7], type=int,
                        help='gpu ids to use')
    parser.add_argument('--lin', action='store_true',
                         help='train linear classifier only')
    parser.add_argument('--generate_cams', action='store_true',
                         help='generate cam')

    args = parser.parse_args()
    config_file = args.config
    device_ids = args.device
    lin_cls = args.lin

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    center = config['MODEL']['img_res'] // 2
    dataloaders = get_dataloaders(config)

    if not args.generate_cams:
        config["MODEL"]["return_att"] = False
        config["CHECKPOINT"]["load_from_checkpoint"] = None
    net = get_model(config, device)

    train_and_evaluate(net, dataloaders, config, device, generate_cams=args.generate_cams)
