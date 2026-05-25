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
