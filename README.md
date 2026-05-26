# PJ Intro

## CLIP-ReID
1. 原论文在clip基础上微调。
    - clip对齐语义信息与图像信息
    - 有text encoder和image encoder
    - text encoder输入prompt，image encoder输入图像
2. 采用两阶段训练：
    - stage1 : 只训练prompt [A photo of a X X X X person.]
    - stage2 : 训练image encoder

### 尝试
1. 在market-1501采用三阶段训练
    - stage1 : 只训练prompt
    - stage2 : 同时训练prompt和image encoder
    - stage3 : 只训练image encoder
    - 结果：
        2026-05-24 03:01:40,160 transreid.train INFO: mAP: 90.1%
        2026-05-24 03:01:40,160 transreid.train INFO: CMC curve, Rank-1  :95.1%
        2026-05-24 03:01:40,160 transreid.train INFO: CMC curve, Rank-5  :98.3%
        2026-05-24 03:01:40,160 transreid.train INFO: CMC curve, Rank-10 :98.9%
    - 较原论文无明显提升
    - 反思：CLIP-ReID 原论文的两阶段设计本质上已经很完整了：stage1 冻结 image/text encoder，只学习每个 ID 的 learnable prompt；stage2 再把学好的 text token 固定住，用它们作为“语义约束”去微调 image encoder。也就是说，原方法的关键不是“多训练 prompt”，而是让 prompt 先形成一个相对稳定的 ID 语义锚点，再约束图像特征学习。原论文也明确说 stage1 只优化 text tokens，stage2 让这些 tokens 和 text encoder 静态化，用来约束 image encoder 微调。三阶段里，stage2 同时训练 prompt 和 image encoder，可能反而削弱了这个固定锚点的作用。因为 prompt 和 image feature 一起动，模型可以通过“两边一起适配”来降低 loss，而不是强迫 image encoder 去靠近一个稳定的文本特征空间。这样 stage2 可能更像普通的 ReID 微调，而不是更强的跨模态约束
    - 数据集相关： Market-1501 本身比较成熟，指标已经接近高位。现在是 mAP 90.1%、Rank-1 95.1%，已经很高了；在这种数据集上，单纯改训练流程通常很难带来明显提升，提升空间可能只有小数点级别，而且容易被随机种子、batch 采样、学习率、epoch 数影响

2. 在market-1501调整lr
    - 第一阶段采用余弦退火，lr_min较小的话，后期更新过慢，增大lr_min
    - 结果的mAP较原论文增大0.1%

3. CCVID上采用两阶段训练，可学习的prompt token增加到6，prompt改为[A X X X X X X person with different clothes.]
    - 结果：
    2026-05-25 11:21:01,153 transreid.test INFO: mAP: 57.8%
    2026-05-25 11:21:01,154 transreid.test INFO: CMC curve, Rank-1  :75.4%
    2026-05-25 11:21:01,154 transreid.test INFO: CMC curve, Rank-2  :76.2%
    2026-05-25 11:21:01,154 transreid.test INFO: CMC curve, Rank-3  :76.8%
    2026-05-25 11:21:01,154 transreid.test INFO: CMC curve, Rank-4  :77.2%
    2026-05-25 11:21:01,154 transreid.test INFO: CMC curve, Rank-5  :77.5%


## TransReID

### 工作内容

1. 跑通原论文 TransReID 的 ViT + SIE + JPM 配置。
   - 使用 `vit_base_patch16_224_TransReID` 作为 backbone。
   - 使用 ImageNet 预训练权重 `jx_vit_base_p16_224-80ecf9dd.pth` 初始化 ViT。
   - 主要训练配置为 `TransReID/configs/AllTrainImgs/vit_transreid_stride.yml`。
   - 训练输入尺寸为 `256x128`，`STRIDE_SIZE=[12, 12]`，开启 `JPM` 和 `SIE_CAMERA`。

2. 新增合并训练集 `all_train_imgs`。
   - 文件：`TransReID/datasets/all_train_imgs.py`
   - 合并 Market1501、DukeMTMC-reID、MSMT17、Occluded_Duke 四个人体 ReID 数据集。
   - 不包含 VeRi 和 VehicleID，因为这两个是车辆 ReID 数据集。
   - 合并时对 pid、camera id、view id 重新编号，避免不同数据集的标签冲突。
   - 四个训练集的 camera 数量合计为 37，因此训练得到的权重中 `base.sie_embed` 的 camera embedding 维度为 37。

3. 新增/整理测试集接入。
   - Market1501：使用原始 `market1501` dataset 类，读取 `query/` 和 `bounding_box_test/`。
   - Move 风格数据：使用 `TransReID/datasets/test_data.py`，读取 `ROOT_DIR/test/<person_id>/*.jpg`。每个身份随机选 1 张作为 query，其余作为 gallery。
   - CCVID：移植 CLIP-ReID 中的 `ccvid.py` 到 `TransReID/datasets/ccvid.py`，读取 `train.txt`、`query.txt`、`gallery.txt`，并将每个 video folder 展开为帧图像。

4. 修改 CCVID 评价逻辑。
   - Market1501 标准评价会过滤同 pid 且同 camid 的 gallery 图像。
   - CCVID 当前适配器中所有样本的 `camid=0`，如果继续使用 Market 过滤规则，会把正确匹配样本过滤掉。
   - 因此在 `TransReID/utils/metrics.py` 中对 `dataset_name == "ccvid"` 特判，不过滤 same-camera positives。

5. 针对无真实 camera 信息的测试集关闭 SIE。
   - Move 和 CCVID 没有可靠的 camera 输入。
   - 测试时使用 `MODEL.SIE_CAMERA False`，不使用训练权重中的 `base.sie_embed`。
   - 加载权重时分类头 shape 不一致会被跳过，这是正常现象；测试阶段只使用特征，不使用分类头。


### 当前结果

使用 `logs/all_train_imgs_stride_20ep/transformer_20.pth`，并在无 camera 信息测试时关闭 SIE：

| Dataset | mAP | Rank-1 | Rank-5 | Rank-10 | Log |
| --- | ---: | ---: | ---: | ---: | --- |
| Market1501 | 72.8% | 87.6% | 95.0% | 96.9% | `TransReID/logs/test_market_no_sie.txt` |
| Move | 50.0% | 81.0% | 91.0% | 94.0% | `TransReID/logs/test_move_no_sie.txt` |
| CCVID | 49.7% | 56.0% | 60.0% | 62.2% | `TransReID/logs/test_ccvid_no_sie.txt` |

### 官方权重参考

TransReID README 中提供了作者训练好的 ReID 权重。当前只下载并测试了官方 Market1501 权重，已复现相同的结果，其他官方权重因与测试集不同，暂未纳入本实验记录。官方 README 中 `TransReID*(ViT)` 的参考指标为：

| Dataset | mAP | Rank-1 |
| --- | ---: | ---: |
| MSMT17 | 67.8% | 85.3% |
| Market1501 | 89.0% | 95.1% |
| DukeMTMC | 82.2% | 90.7% |
| OCC_Duke | 59.5% | 67.4% |


