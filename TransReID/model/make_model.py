import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss


def load_checkpoint_param(module, trained_path):
    param_dict = torch.load(trained_path, map_location='cpu')
    if 'state_dict' in param_dict:
        param_dict = param_dict['state_dict']

    own_state = module.state_dict()
    loaded, skipped = 0, []
    for key, value in param_dict.items():
        key = key.replace('module.', '')
        if key not in own_state:
            skipped.append((key, 'missing in current model'))
            continue
        if own_state[key].shape != value.shape:
            skipped.append((key, 'shape {} vs {}'.format(tuple(own_state[key].shape), tuple(value.shape))))
            continue
        own_state[key].copy_(value)
        loaded += 1

    print('Loading pretrained model from {}'.format(trained_path))
    print('Loaded {} parameters, skipped {}'.format(loaded, len(skipped)))
    for key, reason in skipped:
        print('Skip loading parameter {} ({})'.format(key, reason))


def shuffle_unit(features, shift, group, begin=1):
    """JPM 分支使用的 patch token 重排函数。

    参数含义：
    - features: Transformer 输出 token，形状通常是 [B, 1 + N, C]；
      第 0 个 token 是 cls token，后面 N 个是 patch token。
    - shift: 循环平移的 token 数，用于让局部分块看到不同位置组合。
    - group: shuffle 时切成多少组。
    - begin: patch token 的起始位置，默认跳过第 0 个 cls token。

    返回值仍是 patch token 序列，形状为 [B, N, C]。
    """

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation:
    # 从 begin 开始取 patch token，先做一次循环平移。`begin-1+shift` 这个切点
    # 来自原 TransReID/JPM 实现，用于把局部区域错开。
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation:
    # 先按 group 分组，再交换 group 维和组内位置维，相当于对 patch 顺序做交错重排。
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        # 当 patch 数不能被 group 整除时，复制倒数第二个 token 补齐一个位置。
        # 这是原实现的容错处理，保证下面的 view 可以成功。
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    """用于 backbone 后新增层的 Kaiming 初始化。

    Linear、Conv、BatchNorm 的初始化策略不同：
    - Linear: fan_out，适合分类/投影层；
    - Conv: fan_in，保持前向激活方差；
    - BatchNorm: gamma=1、beta=0，使初始状态接近恒等变换。
    """
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
    """分类器权重初始化。

    ReID 训练中分类头通常用较小标准差的正态分布初始化，避免初始 logits 过大。
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    """ResNet50 版本的 ReID 模型。

    结构流程：
    输入图像 -> ResNet backbone -> 全局平均池化 -> 全局特征 global_feat
    -> 可选 BNNeck -> 分类器/测试特征。

    训练阶段返回 `(cls_score, global_feat)`：
    - cls_score 用于 ID 分类损失；
    - global_feat 通常用于 triplet loss。

    测试阶段根据 `cfg.TEST.NECK_FEAT` 返回 BN 前或 BN 后特征。
    """
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        # 从配置读取 ResNet 和 ReID neck 相关选项。
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            # ResNet50 最后一层输出通道数是 2048。
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            # 加载 ImageNet 预训练权重，只初始化 backbone 部分。
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # 这里保留了 GAP 模块成员，但 forward 中实际使用 functional.avg_pool2d。
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        # ID 分类头：输入 ReID 特征维度，输出训练集身份类别数。
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        # BNNeck：ReID 常用做法。
        # global_feat 用于 triplet loss，BN 后 feat 用于分类/测试，可缓解分类损失和度量损失的目标冲突。
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        # 固定 BN bias，避免 BNNeck 引入可学习平移。
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        # x: [B, 3, H, W]，经过 ResNet 后得到 feature map: [B, 2048, h, w]。
        x = self.base(x)
        # 对空间维 h,w 做全局平均池化，得到每张图的 2048 维全局描述。
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        # neck 控制分类/测试用特征是否经过 BNNeck。
        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                # 如果启用 margin-based 分类层，需要 label 来构造带 margin 的 logits。
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            # 返回 BN 前 global_feat 是为了让 triplet loss 使用原始嵌入空间。
            return cls_score, global_feat
        else:
            # 推理时只返回最终用于检索的特征，不返回分类 logits。
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        load_checkpoint_param(self, trained_path)

    def load_param_finetune(self, model_path):
        # 微调加载：要求 checkpoint 中的参数名和当前模型完全匹配。
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    """Transformer 全局特征版本的 TransReID。

    这个类不使用 JPM 局部分支，只取 Transformer backbone 输出的全局特征。
    支持普通 Linear 分类头，也支持 ArcFace/CosFace/AMSoftmax/CircleLoss 等 margin 分类头。
    """
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        # Transformer 分支主要依赖 cfg.MODEL.TRANSFORMER_TYPE 选择具体 ViT/DeiT 实现。
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        # ViT-Base 默认 hidden dim 是 768；DeiT-Small 会在下面改为 384。
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        # SIE: Side Information Embedding。
        # 当配置关闭 camera/view SIE 时，把数量置 0，backbone 内部就不会构造对应 embedding。
        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        # 通过工厂表实例化具体 Transformer。
        # cam/view 数量、stride、dropout 等都传给 backbone，用于构造 patch embedding 和 SIE。
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            # DeiT-Small 的 embedding dim 是 384，所以分类头和 BNNeck 维度也必须同步调整。
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        # 根据配置选择 ID 分类损失对应的分类头。
        # margin-based head 需要在 forward 时额外传入 label。
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
            # 默认普通 softmax 分类头。
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        # Transformer 输出 global_feat 之后同样接 BNNeck。
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        # backbone 返回每张图的全局 token 特征，形状通常是 [B, C]。
        # cam_label/view_label 用于 SIE；如果配置关闭，backbone 会忽略对应信息。
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        # BN 后特征用于分类头；BN 前特征保留给 triplet loss。
        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                # margin-based 分类头需要 label 才能对正类 logit 加 margin。
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            # 测试阶段根据配置选择 BN 后或 BN 前特征作为检索 embedding。
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        load_checkpoint_param(self, trained_path)

    def load_param_finetune(self, model_path):
        # 与 Backbone 中的微调加载逻辑一致：严格按 key 拷贝。
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    """Transformer + JPM 的局部特征版本。

    JPM（Jigsaw Patch Module）的核心思想：
    1. backbone 先输出完整 token 序列 `[cls, patch_1, ..., patch_N]`；
    2. global branch 用完整 token 序列得到全局特征；
    3. local branch 按 `MODEL.DEVIDE_LENGTH` 把 patch token 分段，每段与 cls token 拼接后再过最后一个 Transformer block；
    4. 训练时同时监督全局特征和多个局部特征；
    5. 测试时把全局特征和局部特征拼接成更长的检索向量。
    """
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        # 与全局 Transformer 一样，按配置决定是否启用 camera/view SIE。
        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        # local_feature=cfg.MODEL.JPM 会让 backbone 返回 token 序列，而不是只返回 cls/global 特征。
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # 复用 backbone 的最后一个 Transformer block 和 norm，复制出两条分支：
        # b1 用于 global branch，b2 用于每个 local branch。
        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        # JPM patch shuffle 相关超参数。
        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        # 将 patch token 均分成多少段；默认是 4，调大后会自动增加局部分支数量。
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        # 是否启用 shuffle_unit 做 patch 重排。
        self.rearrange = rearrange

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        # JPM 默认常见配置是 softmax 分类头；下面仍保留了 margin-based head 的选项。
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
            # 全局分支分类头。
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            # 每个局部分支各自独立分类，强制每个局部区域也具备身份判别能力。
            for i in range(self.divide_length):
                classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
                classifier.apply(weights_init_classifier)
                setattr(self, 'classifier_{}'.format(i + 1), classifier)

        # 全局特征 BNNeck。
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        # 每个局部分支各自使用独立 BNNeck，避免不同局部区域统计量互相干扰。
        for i in range(self.divide_length):
            bottleneck = nn.BatchNorm1d(self.in_planes)
            bottleneck.bias.requires_grad_(False)
            bottleneck.apply(weights_init_kaiming)
            setattr(self, 'bottleneck_{}'.format(i + 1), bottleneck)

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        # local_feature 模式下，features 是完整 token 序列：[B, 1 + N, C]。
        # 第 0 个位置是 cls token，后面是 patch token。
        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch:
        # 对完整 token 序列再过一个复制出来的最后 block + norm，然后取 cls token 作为全局特征。
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch:
        # 去掉 cls token 后得到 patch token 数量，再按 divide_length 计算每段长度。
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        # 保留原始 cls token，后面每个局部分支都会把它拼回局部 patch 前面。
        token = features[:, 0:1]

        if self.rearrange:
            # 训练/测试前可以先对 patch token 做 shift + shuffle，增强局部分支的组合多样性。
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            # 不重排时，直接使用原始 patch token 序列。
            x = features[:, 1:]
        # local branch: 按 divide_length 循环生成多个局部特征。
        local_feats = []
        for i in range(self.divide_length):
            local_feat = x[:, patch_length * i:patch_length * (i + 1)]
            local_feat = self.b2(torch.cat((token, local_feat), dim=1))
            local_feats.append(local_feat[:, 0])

        # 全局和局部特征分别过各自 BNNeck，用于分类头或测试输出。
        feat = self.bottleneck(global_feat)

        local_feat_bns = []
        for i, local_feat in enumerate(local_feats):
            local_feat_bn = getattr(self, 'bottleneck_{}'.format(i + 1))(local_feat)
            local_feat_bns.append(local_feat_bn)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                # 当前分支只为全局特征计算 margin-based 分类 logits。
                # 如果要让局部分支也使用 margin-based loss，需要额外定义局部分类头。
                cls_score = self.classifier(feat, label)
                return cls_score, global_feat
            else:
                # softmax 配置下，全局 + 所有局部分支都参与 ID 分类监督。
                cls_scores = [self.classifier(feat)]
                for i, local_feat_bn in enumerate(local_feat_bns):
                    cls_score = getattr(self, 'classifier_{}'.format(i + 1))(local_feat_bn)
                    cls_scores.append(cls_score)
            # 第一个列表用于 ID loss；第二个列表用于 triplet loss 等度量学习损失。
            return cls_scores, [global_feat] + local_feats  # global feature for triplet loss
        else:
            # 测试时把全局特征和所有局部特征拼接。
            # 局部特征除以 divide_length，用来减小局部分支在拼接向量中的权重。
            if self.neck_feat == 'after':
                return torch.cat([feat] + [local_feat_bn / self.divide_length for local_feat_bn in local_feat_bns], dim=1)
            else:
                return torch.cat([global_feat] + [local_feat / self.divide_length for local_feat in local_feats], dim=1)

    def load_param(self, trained_path):
        load_checkpoint_param(self, trained_path)

    def load_param_finetune(self, model_path):
        # 微调加载：不做 key/shape 容错，要求 checkpoint 与当前模型结构一致。
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


# Transformer 类型到具体构造函数的映射。
__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    """模型构造入口。

    根据配置选择三种结构：
    1. `MODEL.NAME == 'transformer'` 且 `MODEL.JPM == True`：
       构造 Transformer + JPM 局部分支模型；
    2. `MODEL.NAME == 'transformer'` 且 `MODEL.JPM == False`：
       构造只使用全局特征的 Transformer 模型；
    3. 其他情况：
       构造 ResNet50 baseline。
    """
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            # 带 JPM 的版本会输出全局 + 局部特征。
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            # 普通 Transformer 版本只输出全局特征。
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        # 非 transformer 配置默认走 ResNet baseline。
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
