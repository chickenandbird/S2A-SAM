from mmdet.apis import init_detector, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv import Config, DictAction
import torch
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch.distributed as dist
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed

from mmrotate.apis import train_detector
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import (collect_env, get_device, get_root_logger,
                            setup_multi_processes)

# 减少内存碎片（关键！）
torch.cuda.set_per_process_memory_fraction(0.9)  # 限制进程最大占用90%显存
torch.backends.cudnn.benchmark = True      # 加速卷积计算
# 优化内存分配策略
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 2. 加载配置文件
config_path = 'S2A-SAM/configs/s2anet/s2anet_r50_fpn_1x_dota_le135.py'
cfg = Config.fromfile(config_path)

# 3. 修改配置适配新数据集
cfg.data_root = 'instances2/'
cfg.data.train.ann_file = cfg.data_root+'train/labelTxt/'
cfg.data.train.img_prefix = cfg.data_root+'train/images/'
cfg.data.val.ann_file = cfg.data_root+'val/labelTxt/'
cfg.data.val.img_prefix = cfg.data_root+'val/images/'
cfg.data.test.ann_file = cfg.data_root+'val/labelTxt/'
cfg.data.test.img_prefix = cfg.data_root+'val/images/'

# 4. 类别映射与权重处理
# old_classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
#                'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#                'basketball-court', 'storage-tank', 'soccer-ball-field',
#                'roundabout', 'harbor', 'swimming-pool', 'helicopter'] 
old_classes = ['aircraft', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage_tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter']  # 预训练模型类别
new_classes = ['storage_tank','vehicle','aircraft','ship','bridge','sports_facility','roundabout','harbor']  # 新数据集类别
common_classes = list(set(old_classes) & set(new_classes))  # 重叠类别['vehicle', 'ship']

setup_multi_processes(cfg)

# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True

cfg.work_dir = 'log/rbox'

cfg.gpu_ids = range(1)
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
meta = dict()
# log env info
env_info_dict = collect_env()
env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
logger.info('Environment info:\n' + dash_line + env_info + '\n' +
            dash_line)
meta['env_info'] = env_info
meta['config'] = cfg.pretty_text
# log some basic info
logger.info(f'Distributed training: {False}')
logger.info(f'Config:\n{cfg.pretty_text}')

seed = init_random_seed(None)
logger.info(f'Set random seed to {seed}, '
            f'deterministic: {False}')
set_random_seed(seed)
cfg.seed = seed
meta['seed'] = seed

model = build_detector(cfg.model,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
pretrained_weights = torch.load('s2anet_r50_fpn_1x_dota_le135-5dfcf396.pth')['state_dict']

model_state_dict = model.state_dict()
# 定义所有需要处理的分类层关键字（覆盖所有分类头）
cls_keywords = ['cls_convs', 'retina_cls', 'odm_cls']  # 添加缺失的关键字

for key in list(pretrained_weights.keys()):
    # 检查是否为分类层权重（权重或偏置）
    if any(kw in key for kw in cls_keywords) and ('weight' in key or 'bias' in key):
        old_weight = pretrained_weights[key]
        new_weight = model_state_dict[key].clone()
        
        for i, cls_name in enumerate(new_classes):
            if cls_name in common_classes:
                old_idx = old_classes.index(cls_name)
                # 权重切片复制（保持相同类别的权重）
                if 'weight' in key:
                    new_weight[i] = old_weight[old_idx]  # 复制权重矩阵的行
                elif 'bias' in key:
                    new_weight[i] = old_weight[old_idx]  # 复制偏置向量元素
        pretrained_weights[key] = new_weight

# 8. 构建数据集
datasets = [build_dataset(cfg.data.train)]
if len(cfg.workflow) == 2:
    val_dataset = copy.deepcopy(cfg.data.val)
    val_dataset.pipeline = cfg.data.train.pipeline
    datasets.append(build_dataset(val_dataset))



cfg.device = get_device()

# 9. 启动训练
train_detector(model,datasets, cfg, distributed=False, validate=True,timestamp=timestamp,
        meta=meta)
