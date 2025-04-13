import os
from config import cfg                  # 全局配置管理模块
import argparse                         # 命令行参数解析
from datasets import make_dataloader    # 数据加载器构建
from model import make_model            # 模型构建
from processor import do_inference      # 推理流程主函数
from utils.logger import setup_logger   # 日志记录模块


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # 合并配置文件与命令行参数
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)   # 从YAML文件加载配置
    cfg.merge_from_list(args.opts)              # 允许通过命令行覆盖配置项
    cfg.freeze()                                # 冻结配置，防止后续被修改

    # 创建输出目录（用于保存日志）
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化日志记录器（if_train=False表示测试模式）
    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    # 记录配置文件内容
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # 设置可见的GPU设备（根据配置中的DEVICE_ID）
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 构建数据加载器（与训练时类似，但可能关闭数据增强）
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num=make_dataloader(cfg)

    # 构建模型（结构与训练时一致）
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # 加载预训练权重（关键步骤！）
    model.load_param(cfg.TEST.WEIGHT)

    # VehicleID数据集特殊处理（多测试集划分，需10次实验取平均）
    if cfg.DATASETS.NAMES == 'VehicleID':
        # 进行10次独立测试
        for trial in range(10):
            # 重新加载数据加载器（不同测试集划分）
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            # 执行推理，返回Rank-1和Rank-5准确率
            rank_1, rank5 = do_inference(
                cfg,
                model,
                val_loader,
                num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        # 计算10次实验的平均结果
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0))

    # 其他数据集（如Market-1501、DukeMTMC）
    else:
        # 单次推理，直接输出结果
        do_inference(
            cfg,
            model,
            val_loader,
            num_query)

