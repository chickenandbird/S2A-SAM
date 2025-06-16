# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import random
import mmrotate  # noqa: F401
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default='instances2/train/images',help='Image file')
    parser.add_argument('--config',default='s2anet_r50_fpn_1x_dota_le135.py', help='Config file')

    
    # parser.add_argument('--config',default='s2anet_r50_fpn_1x_dota_le135.py', help='Config file')
    parser.add_argument('--checkpoint',default='s2anet_r50_fpn_1x_dota_le135-5dfcf396.pth', help='Checkpoint file')
    # parser.add_argument('--checkpoint',default='log/rbox/epoch_24.pth', help='Checkpoint file')
    parser.add_argument('--out-file', default='/root/autodl-tmp/hhh2', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    # 获取所有标签文件名（不含后缀）
    all_labels = [f.split(".")[0] for f in os.listdir(args.img) if f.endswith(".png")]

    # 随机抽取20个样本的基名
    selected_basenames = random.sample(all_labels, 20)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    print(model.CLASSES)
    # model.CLASSES = ('storage_tank','vehicle','aircraft','ship','bridge','sports_facility','roundabout','harbor')
    model.CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank', 'soccer-ball-field',
            'roundabout', 'harbor', 'swimming-pool', 'helicopter')
    # test a single 
    for i in selected_basenames:
        result = inference_detector(model, os.path.join(args.img, i + ".png"))
        # show the results
        show_result_pyplot(
            model,
            os.path.join(args.img, i + ".png"),
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=os.path.join(args.out_file, i + ".png"))


if __name__ == '__main__':
    args = parse_args()
    main(args)
