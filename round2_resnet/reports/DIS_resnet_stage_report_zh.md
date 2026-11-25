# Galaxy Zoo 形态分类中的投票分布监督

## 研究问题与方法

星系形态可用于研究星系结构和演化。GZ2 为同一图像收集多名志愿者的响应；若只取 argmax，会丢掉分歧信息。分歧可能来自分辨率或信噪比限制、形态处于类别之间，或标注者判断差异，不能仅凭投票熵判断其原因。raw vote fraction 是给定图像上的志愿者响应，不是物理真值。`star or artifact`（class-2）是分类树根节点的筛除类别，不是第三种星系形态。

模型为 ImageNet 预训练 ResNet-18，128 像素 RGB 输入、ImageNet 归一化、batch size 32、Adam、learning rate `3e-4`、weight decay `1e-4`，仅解冻 layer4 和线性分类头。报告使用定稿的展开式 ResNet-18 结构图：竖条表示卷积，弧线表示逐块 shortcut，下方括号标出 stage 和输出尺寸；hard 和 soft 共享架构，只改变监督目标。128 像素来自已有 CPU 吞吐对照（65.72 samples/s；224 像素为 29.44 samples/s），是资源约束下的设计判断，不是对天文图像最优性的证明。八组 soft-only grid 使用 learning rate `1e-4/3e-4`、weight decay `0/1e-4` 和 head-only/layer4+head，分别考察更新尺度、正则化和域适配成本；共同配置按 validation soft CE 选择，calibration discrepancy 和最早 epoch 仅作平局判据。

## 主要证据与解释

固定预算正式评估（three-epoch）的训练规模为 3,100 和 10,000，validation 固定为 750 个对象，自然比例评估池为 10,000 个对象。mean-soft 和 hard-frequency 常数基线的 CE/Brier/accuracy 分别为 `0.7678/0.1676/0.7586` 和 `0.8412/0.1819/0.7586`；模型均值为：

| 训练规模 / 目标 | CE | Brier | accuracy |
|---|---:|---:|---:|
| 3,100 hard | 0.8748 | 0.0880 | 0.8813 |
| 3,100 soft | 0.6156 | 0.0456 | 0.8908 |
| 10,000 hard | 0.8805 | 0.0851 | 0.8892 |
| 10,000 soft | 0.6053 | 0.0384 | 0.9034 |

10,000 规模同时带来约 3.23 倍更新次数，而 class-2 训练对象仍只有 34 个，
因此对象数量和更新预算混杂，不能把规模差异单独归因于数据量。

表格是固定预算正式评估结果；另一个 20-epoch 诊断只用 3,100-object training split、同一 750-object validation 和 seed `20261119`，不构成新的独立评估。在 `lr=3e-4` 下，hard 的训练 hard CE 持续下降而 validation soft CE 从 epoch 1 的 `0.8676` 恶化到 epoch 20 的 `2.1351`；这与训练/验证分离及 hard 目标拟合过强的迹象一致，但不能把原因唯一归于 learning rate，也不能以 loss 绝对值比较 hard 与 soft。soft 的 validation soft CE 在前三个 epoch 后仍改善，epoch 19 为 `0.6251`，epoch 20 为 `0.6290`；`lr=1e-4` 的选择 epoch 为 11，epoch 20 为 `0.6288`。四档学习率（`1e-5`、`1e-4`、`3e-4`、`1e-3`）和正则化结果均属于有限单 seed、20-epoch 诊断，不能证明充分收敛或全局最优。

dropout `0.2` 在新增 seed `20261120` 的 best soft CE 为 `0.6239`，优于 dropout 0 的 `0.6246`，但在 `20261121` 为 `0.6252`，劣于 `0.6236`。筛选 seed `20261119` 不是独立确认；没有一致收益，因此保留 dropout `0`。weight decay `1e-3` 也没有显示一致优势。两种目标统一按 validation soft CE 选模，回答的是投票分布拟合问题；accuracy 是该设定下的次要结果，不是 hard 和 soft 各自以分类准确率充分调优后的比较。

在 10,000 训练规模对应的自然评估池中，按自然池中位数划分的投票熵高、边际低分组为 `H > 0.5462` 且最大/次大投票差 `M <= 0.6897`，共 4,520 个对象。这里 `H` 是归一化投票分布的熵，`M` 是最大与次大投票比例之差。三 seed 均值中，soft 对 hard 的 CE 为 `0.794/1.136`，Brier 为 `0.052/0.133`，accuracy 为 `0.808/0.793`。这说明该分组存在条件性的概率拟合差异，但不能说明投票分歧的物理原因，也不能推出 soft 普遍更好。五个 class-2 对象的诊断中，3,100 规模 hard/soft 为 `5/5/4` 与 `2/2/2`，10,000 规模为 `4/4/5` 与 `3/5/5`；小规模下 soft 识别较少，大规模下平均识别数相同。同一五个对象在各 seed 间重复，不能估计稳定的总体召回率。

GZ2 `extra` 结果应称为 GZ2 内部的 original-to-extra cross-sample generalisation under selection shift，不是跨巡天或已验证的 OOD。它使用冻结评估协议，不属于 20-epoch 诊断。当前结论仍限于单一 ResNet、128 像素和有限训练预算；物理真值校准和因果解释仍未由本研究确定。

## 老师最新意见对应情况

- **已完成实验与解释：** 已纳入真实 ResNet-18 结构图、较长训练曲线、有限学习率和正则化诊断、soft-CE 选模边界，以及投票分歧分组的数值解释。
- **证据受限：** 20 epoch 仍不是充分收敛证明；新诊断仅一个筛选 seed 和两个新增 dropout 确认 seed，独立评估仍对应旧 3-epoch 协议。投票分歧不能被解释为物理真值不确定性。
- **未实施：** 未测试新的 batch size、MLP、其他解冻范围或更广模型搜索；完整数据、权重和环境未随公开仓库分发。

完整结果、配置、预测和图表保存在 `round2_resnet/results/`；报告源文件和结构图分别位于 `round2_resnet/reports/` 与 `round2_resnet/architecture_review/`。
