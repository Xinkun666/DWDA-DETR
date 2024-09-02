# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from data.data_project import xywh_to_rect,draw_rectangles

import numpy as np
import torch

import util.misc as utils




def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # last_index = np.arange(len(data_loader)-1,len(data_loader))
    # image_ids = [1,2,3,4,5,6,8,11]
    # iter = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        # print(targets[0]['image_id'])
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict,idx = criterion(outputs, targets)

        # 可视化训练过程
        # if iter in last_index:
        #     for i,t in enumerate(idx):
        #         boxes = []
        #         labels = []
        #         image_id = targets[i]['image_id'].item()
        #         img_id_str = '{:06d}.jpg'.format(image_id)
        #         for box_idx in t[0]:
        #             box_idx = box_idx.item()
        #             box = outputs['pred_boxes'][i][box_idx]
        #             box_label = outputs['pred_logits'][i][box_idx]
        #             box = box.cpu().detach().numpy()
        #             box_label = box_label.cpu().detach().numpy()
        #             box_label = np.argmax(box_label)
        #             box = xywh_to_rect(box)
        #             orig_size = [targets[i]['orig_size'][0].item(),targets[i]['orig_size'][1].item()]
        #             box = [box[0]*orig_size[0],box[1]*orig_size[1],box[2]*orig_size[0],box[3]*orig_size[1]]
        #             boxes.append(box)
        #             labels.append(box_label)
        #         draw_image = draw_rectangles(None,boxes,labels,img_id=image_id,data_path=args.data_path,mode=args.eval)
        #         save_dir = os.path.join(args.output_dir,str(epoch))
        #         if not os.path.exists(save_dir):
        #             os.makedirs(save_dir)
        #         draw_image.save(os.path.join(save_dir,img_id_str))
        #
        # for i,tar in enumerate(targets):
        #     img_id = tar['image_id'].item()
        #     if img_id in image_ids:
        #         boxes = []
        #         labels = []
        #         img_id_str = '{:06d}.jpg'.format(img_id)
        #         t = idx[i]
        #         for box_idx in t[0]:
        #             box_idx = box_idx.item()
        #             box = outputs['pred_boxes'][i][box_idx]
        #             box_label = outputs['pred_logits'][i][box_idx]
        #             box = box.cpu().detach().numpy()
        #             box_label = box_label.cpu().detach().numpy()
        #             box_label = np.argmax(box_label)
        #             box = xywh_to_rect(box)
        #             orig_size = [targets[i]['orig_size'][0].item(),targets[i]['orig_size'][1].item()]
        #             box = [box[0]*orig_size[0],box[1]*orig_size[1],box[2]*orig_size[0],box[3]*orig_size[1]]
        #             boxes.append(box)
        #             labels.append(box_label)
        #         draw_image = draw_rectangles(None, boxes, labels, img_id=img_id, data_path=args.data_path,
        #                                      mode=args.eval)
        #         save_dir = os.path.join(args.output_dir, str(epoch))
        #         if not os.path.exists(save_dir):
        #             os.makedirs(save_dir)
        #         if not os.path.exists(os.path.join(save_dir,img_id_str)):
        #             draw_image.save(os.path.join(save_dir, img_id_str))

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        iter = iter + 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    file_path = os.path.join(args.output_dir,'training_logs.txt')
    with open(file_path,'a') as f:
        log_info = str(metric_logger)
        f.write(log_info+'\n')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}





@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
