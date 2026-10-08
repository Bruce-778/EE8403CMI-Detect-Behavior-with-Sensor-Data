# 第二名公开方法：受控复现与比较

用户目标：实际训练第二名方法，检查其效果，并与我们的方案比较差异。
本页持续记录实现、运行、失败和实际分数；不以作者成绩替代本项目结果。

## 来源与作者成绩

官方作者仓库：
[Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution](https://github.com/Yamato-Arai/kaggle-cmi-detect-behavior-2nd-place-solution)。
固定源码 commit：`cb53f8d0cc82f33b86403ba35c5a580930475528`。
七个文件的 SHA256 保存于 `configs/second_place_source.json`，统一 CRLF→LF
后校验。源文件放在 Git 忽略的 `outputs/reference_code/second_place/`，下载
入口 `scripts/fetch_second_place_reference.py` 只接受这个 commit 与哈希。
不运行 upstream train/test 的顶层入口，不加载外部未知 checkpoint。

[作者 writeup](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/writeups/2nd-place-solution)
报告的四组线上成绩如下，数值为作者公布的近似值：

| 作者完整 ensemble | Public | Private |
| --- | ---: | ---: |
| 无伪标签、无历史后处理 | 0.862 | 0.854 |
| 在线伪标签、无历史后处理 | 0.865 | 0.858 |
| 无伪标签、有历史后处理 | 0.891 | 0.875 |
| 在线伪标签、有历史后处理 | 0.900 | 0.878 |

已保存的官方最终 Private 排行榜 `outputs/kaggle_submission/private_leaderboard.csv`
明确列出 Rank 2、team daiwakun、**0.878839**。因此该高分有官方排名记录，
不只是 writeup 自报。本项目当前官方 Private 0.833362，相差 0.045477；
这不意味着公开代码任意复跑都会获得同样成绩。

作者最终使用三种 CNN 深度、四种传感器分支、10 folds。公开 test.py 默认
列出四类×10 个本地 checkpoint 路径，仓库没有附权重，也没有自动生成
三架构完整 ensemble 的一键流程。复制源文件不能立即得到上述分数。

## 实测协议

公平比较继续使用我们的固定 **5 folds**，每条 sequence 恰好验证一次，
不改变 validation 样本。模型使用原始公开架构，IMU 15 通道、ToF 320
通道，不使用 THM；每折的联合类别集合仅从训练 IDs 的 orientation、
gesture、initial behavior 拟合，验证真值不参与类别集合拟合。

四分支为 IMU/IMU-zero-rotation/IMU+ToF/IMU+ToF-zero-rotation。
base、simple、deep 原始类均通过可运行性检查。首先运行 base/fold 0 的
四分支，完整 50 epoch，以核对训练、显存与计算预算。这个阶段只有一折，
**不能报告为五折复现成绩**。完整 base 五折需要 20 个模型；三架构五折
ensemble 需要 60 个模型。作者原十折完整配方需要 120 个模型。不同 folds
的成绩不能直接相等；资源允许时另设原始十折协议才能检验该部分配方。

保留 Adam(lr=1e-3, wd=1e-4)、batch=32、50 epoch、10% step warmup + cosine、
phase-aligned Mixup(alpha=0.5)、phase CE 权重 1 和 gradient clip norm 1。
不添加 early stopping；按 upstream test.py 的 **last checkpoint** 评价，
不通过挑选最高 validation epoch 宣称复现。每个 epoch 的官方指标仍记录。

原始 IMU 6D、去重力、角速度来自未修改的 utils.py 与 make_feature_from_np；
保留其固定 dt=1/200、坐标映射与两个特殊 subject 的修正。原方法不拟合
scaler；200 帧尾部截断是公开固定超参数，短 sequence 在右侧 padding。
这与我们默认预处理不同，是参考方法的一部分，生产路径保持原样。

ToF 缺失/-1 填零，没有显式 validity 通道；原 ToF 空间 BatchNorm2d
包含 batch 中的 padding 帧。我们先保留原模型，以后再单独消融这些差异。
phase/orientation/gesture 都只是训练目标，验证 Dataset 返回 target=-1、
phase=-1，模型前向只接受传感器与长度。真实 validation gesture 仅用于评分。

## 必要适配与源代码/说明差异

1. 复用未修改的 model.py、masked_batchnorm.py、utils.py，并只抽取源
   train.py 的特征函数、MixupDataset、collate_fn。训练用原生 PyTorch
   驱动等价损失/优化步骤，省去 Lightning、WandB 等新增依赖。框架、
   dataloader workers、GPU/运行库和随机数实现差异仍可能影响数值。
2. 原 GestureDataset 取样时对缓存数组原地置零/翻转；我们每次读取先复制，
   特殊 subject 修正在缓存构建时只做一次。否则 dropout 与重复读取会
   改变底层样本。该修复明确记录，不把适配结果称为逐字运行的原始分数。
3. writeup 说 move phase 末端对齐；固定 commit 的 Mixup 实现是 phase
   分段后的前部截断/补齐。复现使用该 commit 的实际函数，不自行改成
   说明中的末端对齐。后续可做代码版与描述版的独立对照。
4. 实际 base IMU 有三个 stems：acc、6D+angular velocity、linear acc。
   它与 writeup 中列出的四组独立处理并不完全相同；以代码为准。
5. 原推理先平均联合 logits，再取联合 argmax 映射为 gesture。我们的
   复现保持这个规则。保存的 18 类 marginal probabilities 用于诊断，
   其 argmax 可能与联合 argmax 映射不同，不混淆两种决策规则。

## 与我们已上线方法的差异

| 项目 | 我们当前 cnn_dynamics_mixup | 第二名代码参考 |
| --- | --- | --- |
| IMU | 34D，包含运动变化/相关特征 | 15D 物理特征 |
| 缺失路由 | IMU/全模态两个分支 | 额外拆分旋转缺失，共四分支 |
| ToF | 每 sensor 2×2 有效区域统计、时间 1D CNN | 8×8 逐帧 Conv2d，再时间 1D CNN |
| THM | 使用 | 不使用 |
| 归一化 | 训练 fold 的 masked scaler | 无额外数据 scaler，模型 BatchNorm |
| 长度/padding | 训练 fold 95% 分位数，左 padding | 固定最大 200，右 padding |
| 聚合 | masked mean/max | 预测阶段概率加权的三个 attention |
| 分类 | CE18 + 0.5 CE9 + 0.1 BCE2 | 联合类别 CE + phase CE |
| Mixup | 整体序列混合 | 按三个阶段分别混合 |
| 学习率计划 | ReduceLROnPlateau，early stopping | warmup/cosine，50 epoch 固定结束 |
| ensemble | 单架构 A/B 五折 | 三架构、四分支、原始十折 |
| history/pseudo | 当前上线均无 | 两项独立推理阶段增强 |

比较会先隔离“无历史/无伪标签”的模型能力，再分别添加联合类别历史
分配和在线伪标签。历史只能读取当前 subject 已到达的数据，报告三个
固定输入顺序；伪标签更新禁止使用验证真值，验证 subject 不可反向进入
训练 fold。两者属于不同信息协议，单独报告，不能归因于网络架构。

原始输入、固定 aux_dropout50、IMU-only 复用已有 sequence-ID 哈希遮挡，
保存逐折指标、完整 OOF 和标准差。原方法只用 ToF 决定辅助分支，我们
当前方法按 THM/ToF 可用性路由；记录这一差异，不删除不适合该路由的样本。

## 运行与现状

```powershell
python -s scripts/fetch_second_place_reference.py
python -s -m unittest discover -s tests -p test_second_place.py -v
python -s scripts/export_second_place_training.py
# 本地有 CUDA 时可执行完整基础架构五折：
python -s -u scripts/train_second_place.py --device cuda --folds 0 1 2 3 4 --output outputs/second_place/base_fivefold_v1
# 三架构完整五折另设独立输出：
python -s -u scripts/train_second_place.py --device cuda --folds 0 1 2 3 4 --architectures base simple deep --output outputs/second_place/ensemble_fivefold_v1
```

本机 torch 2.8.0+cpu，没有 CUDA。五项新检查通过，覆盖六个原始架构的
forward/backward、验证 phase 不影响预测、train-only 类别集合、缓存不变、
原始物理特征缓存与真实训练/checkpoint/held-out 输出闭环。小样本 smoke
分数不作为比赛数据成绩。

GPU Notebook 已导出，嵌入来源哈希及原固定 folds，所有代码 cell 语法
检查通过，嵌入的训练脚本/源码/folds 与本地逐字一致。默认运行 base、
fold 0、四分支各 50 epoch；关闭互联网，不连接 WandB，不读取隐藏 test
标签，不创建比赛提交。模型/逐折预测打包到 `second_place_experiments.zip`，
不打包大型特征缓存；失败时保留已完成的部分，并明确标为不完整。

实际云端版本、运行状态、真正的验证分数会记入本页和
`experiments/results/second_place_reproduction.json`；当前没有新的复现分数。
