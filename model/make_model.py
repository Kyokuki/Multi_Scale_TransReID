import torch
import torch.nn as nn
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_base_patch32_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss


def weights_init_kaiming(m):
    """Kaiming初始化（适用于卷积、全连接层）"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    """分类层初始化（全连接层，标准差0.001）"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class BuildTransformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(BuildTransformer, self).__init__()
        # 配置参数解析
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        print('-----------------------------------------------------------------')
        print('using Transformer_name: !!! {} !!!'.format(cfg.MODEL.NAME))
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN,
                                                        sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num,
                                                        view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        print('-----------------------------------------------------------------')
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
            
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class BuildTransformerMulti(nn.Module):
    """标准Transformer模型"""
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(BuildTransformerMulti, self).__init__()
        # 配置参数解析
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path_large = cfg.MODEL.PRETRAIN_PATH_LARGE
        model_path_small = cfg.MODEL.PRETRAIN_PATH_SMALL
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        print('-----------------------------------------------------------------')
        print('using Large_Scale_Transformer: !!! {} !!!'.format(cfg.MODEL.NAME_LARGE))
        self.base_large = factory[cfg.MODEL.TRANSFORMER_TYPE_LARGE](
                                                            img_size=cfg.INPUT.SIZE_TRAIN,
                                                            sie_xishu=cfg.MODEL.SIE_COE,            # SIE系数（控制相机/视角嵌入强度）
                                                            camera=camera_num,                      # 相机数量
                                                            view=view_num,                          # 视角数量
                                                            stride_size=cfg.MODEL.STRIDE_SIZE_LARGE,    # 步长（重叠分块）
                                                            drop_path_rate=cfg.MODEL.DROP_PATH,     # DropPath概率
                                                            drop_rate=cfg.MODEL.DROP_OUT,
                                                            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        print('-----------------------------------------------------------------')
        print('using Small_Scale_Transformer: !!! {} !!!'.format(cfg.MODEL.NAME_SMALL))
        self.base_small = factory[cfg.MODEL.TRANSFORMER_TYPE_SMALL](
                                                            img_size=cfg.INPUT.SIZE_TRAIN,
                                                            sie_xishu=cfg.MODEL.SIE_COE,            # SIE系数（控制相机/视角嵌入强度）
                                                            camera=camera_num,                      # 相机数量
                                                            view=view_num,                          # 视角数量
                                                            stride_size=cfg.MODEL.STRIDE_SIZE_SMALL,    # 步长（重叠分块）
                                                            drop_path_rate=cfg.MODEL.DROP_PATH,     # DropPath概率
                                                            drop_rate=cfg.MODEL.DROP_OUT,
                                                            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        print('-----------------------------------------------------------------')

        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384

        # 加载ImageNet预训练权重
        if pretrain_choice == 'multi_imagenet':
            self.base_large.load_param(model_path_large)
            self.base_small.load_param(model_path_small)
            print('Loading pretrained ImageNet model(LARGE)......from {}'.format(model_path_large))
            print('Loading pretrained ImageNet model(SMALL)......from {}'.format(model_path_small))

        self.gap = nn.AdaptiveAvgPool2d(1)

        # 分类器与损失函数配置
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        # 特征归一化层（BNNeck）
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):

        global_feat_small = self.base_small(x, cam_label=cam_label, view_label=view_label)
        global_feat_large = self.base_large(x, cam_label=cam_label, view_label=view_label)
        feat_small = self.bottleneck(global_feat_small)
        feat_large = self.bottleneck(global_feat_large)

        # feat_fusion = torch.stack([feat_small, feat_large], dim=0)
        # feat_fusion = feat_fusion.permute(1,0,2)
        # feat_fusion, _ = self.attention(feat_fusion, feat_fusion, feat_fusion)
        # feat_fusion = feat_fusion[:, 0]

        # # 训练时计算分类损失
        # if self.training:
        #     if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        #         cls_score = self.classifier(feat_fusion, label)
        #     else:
        #         cls_score = self.classifier(feat_fusion)

        #     return cls_score, feat_fusion

        # # 测试时返回特征
        # else:
        #     if self.neck_feat == 'after':
        #         # print("Test with feature after BN")
        #         return feat_fusion
        #     else:
        #         # print("Test with feature before BN")
        #         return feat_fusion

        # 训练时计算分类损失
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score_small = self.classifier(feat_small, label)
                cls_score_large = self.classifier(feat_large, label)
            else:
                cls_score_small = self.classifier(feat_small)
                cls_score_large = self.classifier(feat_large)

            return [cls_score_small, cls_score_large], [global_feat_small, global_feat_large]

        # 测试时返回特征
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat_small, feat_large], dim=1)
            else:
                # print("Test with feature before BN")
                return torch.cat([global_feat_small, global_feat_large], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, weights_only=False)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for fine-tuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,

    'vit_base_patch32_224_TransReID': vit_base_patch32_224_TransReID
}


def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        model = BuildTransformer(num_class, camera_num, view_num, cfg, __factory_T_type)
        print('===========building transformer===========')
    if cfg.MODEL.NAME == 'multi_scale_transformer':
        model = BuildTransformerMulti(num_class, camera_num, view_num, cfg, __factory_T_type)
        print('===========building multi scale transformer===========')
    return model
