import torch
import torchvision.transforms as T                  # 图像预处理库
from torch.utils.data import DataLoader             # PyTorch数据加载器

from .bases import ImageDataset# 自定义基础数据集类
from timm.data.random_erasing import RandomErasing  # 随机擦除数据增强
from .sampler import RandomIdentitySampler          # 三元组采样器（单卡）
from .sampler_ddp import RandomIdentitySampler_DDP  # 分布式三元组采样器
import torch.distributed as dist                    # 分布式训练支持

from .dukemtmcreid import DukeMTMCreID              # DukeMTMC-reID数据集
from .market1501 import Market1501                  # Market-1501数据集
from .msmt17 import MSMT17                          # MSMT17数据集
from .occ_duke import OCC_DukeMTMCreID              # Occluded-DukeMTMC数据集
from .vehicleid import VehicleID                    # VehicleID数据集
from .veri import VeRi                              # VeRi数据集


# 数据集名称到对应类的映射（工厂模式）
__factory = {
    'market1501':   Market1501,
    'dukemtmc':     DukeMTMCreID,
    'msmt17':       MSMT17,
    'occ_duke':     OCC_DukeMTMCreID,
    'veri':         VeRi,
    'VehicleID':    VehicleID,
}


def train_collate_fn(batch):
    """
        整理训练批次数据，处理数据增强后的样本：
        - imgs: 图像张量列表
        - pids: 行人ID列表
        - camids: 相机ID列表
        - viewids: 视角ID列表
        - _: 占位符（原始代码中可能包含其他信息，如路径）
    """

    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,


def val_collate_fn(batch):
    """
        整理验证/测试批次数据，保留图像路径供调试：
        - img_paths: 图像路径列表（用于可视化或错误分析）
    """
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
    """根据配置构建训练和验证数据加载器"""
    # ------------------------- 数据增强定义 -------------------------
    # 训练集增强：Resize、翻转、填充、裁剪、归一化、随机擦除
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),    # 双三次插值
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),           # 随机水平翻转概率
            T.Pad(cfg.INPUT.PADDING),                           # 填充（为随机裁剪做准备）
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),                 # 随机裁剪至目标尺寸
            T.ToTensor(),                                       # 转为Tensor
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),    # 归一化
            RandomErasing(                                      # 随机擦除（模拟遮挡）
                probability=cfg.INPUT.RE_PROB,
                mode='pixel',
                max_count=1,
                device='cpu'),
        ])

    # 验证集增强：仅Resize和归一化
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),      # 调整到测试尺寸
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    # ------------------------- 数据集加载 -------------------------
    num_workers = cfg.DATALOADER.NUM_WORKERS    # 数据加载子进程数

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)     # 根据名称实例化数据集

    # 封装训练集（应用数据增强）
    train_set = ImageDataset(dataset.train, train_transforms)       # dataset.train为训练样本列表
    train_set_normal = ImageDataset(dataset.train, val_transforms)  # 无增强版本（用于某些评估）

    # 获取元信息
    num_classes = dataset.num_train_pids    # 训练集行人ID总数
    cam_num = dataset.num_train_cams        # 相机数量（用于SIE模块）
    view_num = dataset.num_train_vids       # 视角数量（用于SIE模块）

    # ------------------------- 训练数据加载器 -------------------------
    # 根据采样器类型选择数据加载策略
    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:    # 分布式训练
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()     # 单卡批次大小
            data_sampler = RandomIdentitySampler_DDP(   # 分布式身份采样器
                dataset.train,
                cfg.SOLVER.IMS_PER_BATCH,
                cfg.DATALOADER.NUM_INSTANCE)            # 每个ID采样的实例数
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                data_sampler,
                mini_batch_size,
                True)   # 是否丢弃不完整批次
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,    # 分布式批次采样
                collate_fn=train_collate_fn,    # 整理函数
                pin_memory=True,                # 锁页内存加速传输
            )
        else:   # 单卡训练
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(  # 普通身份采样器
                    dataset.train,
                    cfg.SOLVER.IMS_PER_BATCH,
                    cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':   # 简单随机采样
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=True,   # 随机打乱
            num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    # ------------------------- 验证数据加载器 -------------------------
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
