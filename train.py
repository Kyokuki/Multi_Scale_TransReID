from utils.logger import setup_logger   # 日志记录模块
from datasets import make_dataloader    # 数据加载器构建
from model import make_model            # 模型构建
from solver import make_optimizer       # 优化器构建
from solver.scheduler_factory import create_scheduler   # 学习率调度器
from loss import make_loss              # 损失函数构建
from processor import do_train          # 训练流程主函数
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg                  # 全局配置管理


def set_seed(seed):
    """
    设置随机种子（确保实验可复现性）
    """
    torch.manual_seed(seed)             # 设置PyTorch随机种子
    torch.cuda.manual_seed(seed)        # 设置当前GPU的随机种子
    torch.cuda.manual_seed_all(seed)    # 设置所有GPU的随机种子（多卡训练）
    np.random.seed(seed)                # 设置NumPy随机种子
    random.seed(seed)                   # 设置Python随机种子
    torch.backends.cudnn.deterministic = True   # 确保CUDA卷积运算结果确定
    torch.backends.cudnn.benchmark = True       # 启用CUDA卷积优化（提升速度）


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    # 合并配置文件与命令行参数
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)   # 从YAML文件加载配置
    cfg.merge_from_list(args.opts)  # 允许通过命令行覆盖配置项
    cfg.freeze()                    # 冻结配置，防止后续被修改

    # 设置随机种子（确保实验可重复）
    # set_seed(cfg.SOLVER.SEED)

    # 如果是分布式训练，设置当前GPU设备
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    # 创建输出目录（保存模型和日志）
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化日志记录器
    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    # 记录配置文件内容
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # 初始化分布式训练进程组
    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # 设置可见的GPU设备（根据配置中的DEVICE_ID）
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 构建数据加载器
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # 构建模型
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # 构建损失函数
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # 构建优化器
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    # 创建学习率调度器
    scheduler = create_scheduler(cfg, optimizer)

    # 调用训练主函数
    do_train(
        cfg,                # 全局配置
        model,              # 模型实例
        center_criterion,   # 中心损失（可选）
        train_loader,       # 训练数据加载器
        val_loader,         # 验证数据加载器
        optimizer,          # 主优化器
        optimizer_center,   # 中心损失优化器
        scheduler,          # 学习率调度器
        loss_func,          # 主损失函数
        num_query,          # 验证集查询数（计算mAP/Rank-1）
        args.local_rank     # 分布式训练的本地GPU编号
    )
