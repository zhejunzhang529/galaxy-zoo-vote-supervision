# DIS round-2 teacher brief

已完成：补充 GZ2 形态研究动机、投票分歧的可能来源、ResNet 局部特征与
ImageNet 迁移的设计解释；记录 128/224 吞吐、正式训练时间、标签/图像
规模和各脚本的图像读取边界。原有 8 组 soft-only grid、3 个 paired seed、两种
训练规模 hard/soft 实验和冻结 checkpoint 保持不变。

正式评估仍是原 three-epoch 协议：两种训练规模、hard/soft 目标、三个
paired seed，随后使用冻结的独立评估池。证据入口为
`round2_resnet/configs/formal.json`、`round2_resnet/results/formal_runs.json`
和 `round2_resnet/results/formal_summary.json`。

另已完成 train/validation-only 的 20-epoch 诊断：训练预算记录在
`round2_resnet/configs/training_budget_diagnostic.json` 与
`round2_resnet/results/closeout/training_budget_diagnostic_runs.json`；四档
学习率在 `round2_resnet/configs/learning_rate_endpoints.json`、
`round2_resnet/src/train_learning_rate_endpoints.py` 及
`round2_resnet/results/closeout/learning_rate_endpoints/`；有限正则化在
`round2_resnet/configs/regularization_diagnostic.json`、
`round2_resnet/src/train_regularization_diagnostic.py` 及
`round2_resnet/results/closeout/regularization_diagnostic/`；新增 dropout
seeds 在 `round2_resnet/configs/dropout_multiseed_confirmation.json`、
`round2_resnet/src/train_dropout_multiseed_confirmation.py` 及
`round2_resnet/results/closeout/dropout_multiseed_confirmation/`。这些诊断
不读取独立评估池。两个新增 seed 没有显示一致的 dropout 收益，因此保留
dropout=0 作为参考配置；这不是新的正式 hard/soft 多 seed 比较。

已明确 validation soft CE 是共同选模标准，校准差异和最早 epoch 只作
平局判据；因此 accuracy 是该规则下的结果，不是分别按分类准确率充分调优
后的比较。五对象 class-2 诊断和 extra 的 selection-shift 检验仍按原边界
解释，不能宣称 OOD、跨巡天泛化或物理真值校准。

发布状态：清理后的公开快照已发布至
`https://github.com/zhejunzhang529/galaxy-zoo-vote-supervision`。完整数据、
权重和环境未随仓库分发。

extra 检验继续使用固定 1,000 对象和六个冻结 checkpoint。六次
forward-only 计时合计约 28.20 秒；包含加载、指标和 bootstrap 的外围循环
为 31.62 秒；小批量边界估计总成本约 48.5 秒。extra class-2 只有 4 个
对象，逐类结果仅作描述，不能宣称 OOD、跨巡天泛化或物理真值校准。

已有结果仍限定在 soft-only grid 选择出的共同配置。RGB Logistic 补充基线
的有效协议为 maxiter=1000；旧的 80/300 次未收敛尝试保留为历史无效记录。

请老师指导：

1. 下一轮资源应优先用于更长训练预算，还是自然比例且含足够 class-2 的评估池。
2. raw vote fraction 不是物理真值时，当前 CE/Brier 是否足够作为主要概率目标。
3. 如何设计能区分训练对象数与更新预算影响的数据量对照。
