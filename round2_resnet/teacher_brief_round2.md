# DIS 阶段教师简报（本轮）

## 已完成

- 将研究动机改为围绕 GZ2 投票分歧：raw vote fraction 是志愿者响应而非物理真值；分歧可能来自观测限制、类别边界和标注差异，投票熵本身不能判定原因。
- 用已核对的 ResNet-18 结构图替换通用流程图，明确 128 像素输入、冻结早期层、layer4+classifier 微调，以及 hard/soft 共用同一架构。
- 将 20-epoch train/validation-only 诊断纳入解释：hard training CE 下降而 validation soft CE 恶化；soft validation soft CE 在前三个 epoch 后仍有改善但存在波动。预算、四档 learning rate、有限正则化和新增 dropout seeds 的配置、源码和结果分别位于 `configs/`、`src/train_*.py` 和 `results/closeout/` 的对应路径。
- 明确 hard/soft 都按 validation soft CE 选 checkpoint，因此主要比较的是投票分布拟合；accuracy 不是分别充分优化后的主目标。
- 纳入一个有数字支持的高熵、低投票边际分组结果：旧 3-epoch 正式自然池中 n=4,520，soft/hard 均值 CE 为 0.794/1.136、Brier 为 0.052/0.133、accuracy 为 0.808/0.793。该结果是条件性比较，不解释投票分歧的物理原因。
- 修正 class-2 诊断结论：小规模下 soft 识别数较少，大规模下平均识别数相同；同一五个对象跨 seed 重复，不能估计总体召回率。

## 证据边界

旧的 3-epoch 训练、三 paired seed 和独立评估结果均保留不变；20-epoch 结果只来自 small_3100 的 train/validation，不构成延长训练后的独立评估。dropout=0.2 在两个新增 seed 上没有一致改善，继续保留 dropout=0 参考配置；seed 20261119 只用于筛选，不能算独立确认。GZ2 `extra` 仅称为 original-to-extra cross-sample generalisation under selection shift，不称跨巡天 OOD。

## 老师意见状态

- **已实施：** 结构图、训练曲线、soft-CE 选模解释、投票分歧分组分析、有限学习率/正则化诊断及一页正文整合。
- **证据受限：** 20 epoch 不能证明充分收敛；投票分歧原因和物理真值校准仍不可由当前数据确定。
- **未实施：** 未测试 batch size、MLP、其他解冻范围或更宽模型搜索；20-epoch 诊断不是延长训练后的独立评估。
- **发布状态：** 清理后的公开快照已发布至 `https://github.com/zhejunzhang529/galaxy-zoo-vote-supervision`；GitHub 仓库不包含原始数据、权重或环境。

请老师指导：

1. 下一步资源应优先投入更长训练预算，还是投入自然比例且含足够 class-2 的独立评估池。
2. 在 raw vote fraction 不是物理真值的前提下，CE/Brier 是否适合作为主要概率目标。
3. 如何设计实验，区分训练对象数量与更新预算对结果的影响。
