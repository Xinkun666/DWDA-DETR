import argparse
import datetime
import json
import os
import os.path
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
from engine import evaluate, train_one_epoch
from models.detr import build
from data.data_project import build_dataset
import torch.nn.functional as F
from data.data_project import xywh_to_rect,draw_rectangles
from thop import profile
from util.misc import NestedTensor
from PIL import Image, ImageDraw

def get_args_parser():
    parser = argparse.ArgumentParser('DAT_DETR with SwinTransformer', add_help=False)
    # * weight setting
    parser.add_argument('--weight',default='output/20240311/10/checkpoint0199.pth',type=str)

    # * data setting
    parser.add_argument('--data_path', default='evaluate/images3', type=str)
    parser.add_argument('--dataset_file',default='fish',type=str)

    parser.add_argument('--num_workers', default=1, type=int,help='num of workers to use')


    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--output_dir',default='./output/20240313',help='the path to save model pth')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)


    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--positional_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=768, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=2, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--aux_loss', default=False, type=bool)

    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    parser.add_argument('--two_stage', default=False, action='store_true')

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=0, type=float)
    parser.add_argument('--ciou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # * Segmentation
    parser.add_argument('--masks', action='store_true', default=False,
                        help="Train segmentation head if the flag is provided")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=0, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_ciou', default=2, type=float,
                        help="ciou box coefficient in the matching cost")

    return parser

def count_folders(directory):
    # 获取目录中的所有项目
    all_items = os.listdir(directory)

    # 使用列表推导式筛选出文件夹
    folders = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]

    # 返回文件夹的数量
    return len(folders)

def main(args):
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset = build_dataset(image_set='test', args=args)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = DataLoader(dataset, 1, sampler=sampler,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    output_dir = Path(args.output_dir)

    checkpoint = torch.load(args.weight, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print("Start Detect")
    start_time = time.time()


    for samples, img_id in data_loader:
        samples = samples.to(device)

        # flops, params = profile(model, inputs=(samples,), verbose=False)
        # print(f'flops : {flops},parsms : {params}')

        outputs = model(samples)
        outputs_classes = F.softmax(outputs['pred_logits'],dim=2)
        # 首先将tensor移动到CPU，然后转换为NumPy数组
        outputs_boxes = outputs['pred_boxes'].cpu().detach().numpy()
        outputs_classes = outputs_classes.cpu().detach().numpy()
        outputs_boxes = outputs_boxes.reshape(args.num_queries,4).tolist()
        outputs_classes = outputs_classes.reshape(args.num_queries,3).tolist()

        boxes = []
        labels = []
        confidence = []

        boxes_head,labels_head,confidence_head = [],[],[]
        boxes_tail,labels_tail,confidence_tail = [],[],[]

        for i,l in enumerate(outputs_classes):
            if l[np.argmax(l)] > 0.9:
                if np.argmax(l) == 1:
                    labels_head.append(np.argmax(l))
                    boxes_head.append(xywh_to_rect(outputs_boxes[i]))
                    confidence_head.append(l[np.argmax(l)])
                elif np.argmax(l) == 2:
                    labels_tail.append(np.argmax(l))
                    boxes_tail.append(xywh_to_rect(outputs_boxes[i]))
                    confidence_tail.append(l[np.argmax(l)])
        if confidence_head != []:
            if len(confidence_head)>1:
                boxes.append(boxes_head[np.argmax(confidence_head)])
                labels.append(labels_head[np.argmax(confidence_head)])
                confidence.append(labels_head[np.argmax(confidence_head)])
            else:
                boxes.extend(boxes_head)
                labels.extend(labels_head)
                confidence.extend(confidence_head)
        if confidence_tail != []:
            if len(confidence_tail) > 1:
                boxes.append(boxes_tail[np.argmax(confidence_tail)])
                labels.append(labels_tail[np.argmax(confidence_tail)])
                confidence.append(labels_tail[np.argmax(confidence_tail)])
            else:
                boxes.extend(boxes_tail)
                labels.extend(labels_tail)
                confidence.extend(confidence_tail)
        draw_image = draw_rectangles(None,boxes,labels,img_id=img_id[0],data_path=args.data_path,mode=2)
        img_id_str = '{:06d}.jpg'.format(img_id[0])
        save_dir = os.path.join(args.output_dir, img_id_str)
        draw_image.save(save_dir)
        print(f'{img_id_str}已检测完成，保存到{args.output_dir}目录中\n')

        # detection txt dir
        detect_dir = './evaluate/detections/4'
        if not os.path.exists(detect_dir):
            os.makedirs(detect_dir)
        txt_str = img_id_str.replace('jpg','txt')
        txt_dir = os.path.join(detect_dir,txt_str)
        #记录检测数据计算指标
        with open(txt_dir, 'w') as file:
            for label,conf,box in zip(labels,confidence,boxes):
                file.write(f'{label} {conf} {box[0]} {box[1]} {box[2]} {box[3]}\n')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Detect format time {format(total_time_str)}, time{total_time}')




if __name__ == '__main__':

    parser = argparse.ArgumentParser('DAT_DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir,'detect')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    counter = count_folders(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, str(counter))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)