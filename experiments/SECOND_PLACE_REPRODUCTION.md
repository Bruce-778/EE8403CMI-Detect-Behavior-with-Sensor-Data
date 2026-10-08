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

### GPU 首次运行：版本 1，已被 Kaggle 接受

[CMI Second Place Fixed Fold Reproduction v1 / 356304653](https://www.kaggle.com/code/mingweiwei03/cmi-second-place-fixed-fold-reproduction?scriptVersionId=356304653)。
页面明确显示 PRIVATE、Version #1 with GPU T4 x2、Running；官方 CMI
比赛输入已挂载、Internet off。开始时 GPU 剩余额度约 27 小时。
这里只启动 base/fold 0 的四分支各 50 epoch，尚无实际训练完成分数。
78 项全套测试通过（58.725 秒），Notebook 嵌入源码和固定 folds 已核对。
实现与设计本地 commit `feedb82`，不 push、不提交比赛。

### 运行与结果回收核验：2026-10-08 07:01 UTC

GPU 日志已确认原始 train/demo 的 SHA256 与本地固定输入一致。实际环境
为 Tesla T4、torch 2.11.0+cu128；完成全部 8,151 个 sequence 的特征缓存，
base/IMU 已进入 50 epoch 训练。首个模型每 epoch 约 24 秒，这是局部实测，
不据此推断 ToF 分支或三架构完整复现的总耗时。最终分数仍待完成和核验。
页面 stdout 行重复展示，启动脚本只有一次训练调用，不能当作两次实验。

新增结果核验脚本 `scripts/evaluate_second_place.py`：检查输入哈希、实际
train/validation IDs 与 subject、仅训练集建立的联合类别、50 epoch history、
最后 checkpoint、联合 logits 与保存的概率/决策一致性，重建四分支路由，
对照原方法的相同 held-out sequence。五折齐全才产生均值、样本标准差和
完整 OOF；只有首折时明确记为筛选。联合类别历史分配与上游 Hungarian
函数逐个 prefix 等价的检查通过；仅返回新到达样本，既有返回值不回改。
超过联合类别数时记录 baseline fallback 并保留样本，不删除困难样本。

发现并处理结果文件的行顺序风险：原训练 logits 按缓存顺序保存，预测
CSV 则排序。后续运行显式保存 `joint_sequence_ids.npy`、顺序 provenance，
ZIP 包含小型 cache metadata/provenance，但仍排除大型特征数组。正在运行
的 v1 嵌入的是旧脚本，以上更改不会改变其训练配置或输出。回收 v1 必须
同时取得该版本的 `cache/metadata.csv`，按 ID 对齐并检查 logits/CSV；禁止
假定原顺序等于排序。结果脚本缺少该文件时会拒绝评价。

```powershell
& 'D:\anaconda\envs\cmi\python.exe' -s scripts/evaluate_second_place.py `
  --archive '<下载的 second_place_experiments.zip>' `
  --metadata '<同版本 cache/metadata.csv>' `
  --output-dir outputs/kaggle_training/imported/second_place_base_fold0_v1 `
  --name second_place_base_fold0_v1 `
  --source-url 'https://www.kaggle.com/code/mingweiwei03/cmi-second-place-fixed-fold-reproduction?scriptVersionId=356304653'
```

六项第二名相关测试通过（4.982 秒），包括逆序 logits 的对齐、真实小样本
训练/权重/概率/metric 核验、training subject 被篡改时拒绝、原始历史函数
prefix 对照及 overflow。小样本分数不作为真实比赛复现结果。

### 首个完整训练分支日志：2026-10-08 00:19 PDT

v1 仍为 Running。base/imu fold 0 已训练到 50/50，日志显示最后 epoch 的
CMI **0.832582**、loss 1.7911，训练 6,524 / 验证 1,627 个 sequence，联合
类别 102、参数 3,459,948。使用最后 epoch，不能用更高的 epoch 49 分数
替代原始 checkpoint 规则。base/imu_rot 正在训练，其余两个 ToF 分支尚未
开始。运行过程没有重新启动、修改训练或提交比赛。

重新读取冻结原方法 IMU 的相同 fold 0 OOF，官方指标为
**0.7756521483129168**，因此相对上述六位小数日志的初步差为约 **+0.05693**。
这只是单折日志提示，训练结果 ZIP、真实训练 subject 与 OOF 尚未下载
核验，不能确认完整五折提升，不能称为线上分数。此项保存于 JSON 的
`pending_artifact_audit`；正式 `actual_reproduction_scores` 仍为 null。
日志证据存于 `outputs/second_place/evidence/v1_imu_completed_log.txt`。

### 旋转缺失分支完成：2026-10-08 00:33 PDT

base/imu_rot fold 0 完成 50/50，最后 epoch 日志 CMI **0.744109**、loss
2.3924；同样为 6,524 train / 1,627 validation、102 联合类别。该分支把
全部 sequence 的旋转及其衍生通道置零，仅在推理的旋转缺失路由使用。
信息条件与原始 observed IMU 不同，不用 0.744109 与 0.832582 的差来判断
模型进步或退步。完整路由仍需另两个 ToF 分支完成，且下载真实 artifact
核验；此分支记录也属于 pending_artifact_audit。

base/all (IMU + ToF，不含 THM) 已开始，参数 7,101,852，首个 epoch 实测
136.7 秒，相比 IMU 约 24 秒更慢。此数为启动 epoch 耗时，后续应测稳定
耗时并检查实际剩余 GPU quota，再安排完整五折和三架构，不按 IMU 耗时
估算完整复现、不提前宣称有足够算力。当前 v1 正常继续，不重复训练，
没有修改超参或创建线上比赛提交。证据文件
`outputs/second_place/evidence/v1_imu_rot_completed_log.txt`。

### IMU + ToF 分支完成：2026-10-08 02:34 PDT

base/all fold 0 完成 50/50，最后 epoch 日志 CMI **0.888491**、loss 1.6554；
6,524 train / 1,627 validation、102 联合类别、7,101,852 参数。首个基础
ToF 模型的 50 次 epoch 耗时总计约 6,788.3 秒（按日志舍入值求和），排除
首个 epoch 后中位耗时 **135.8 秒/epoch**。完整复现预算须按实际 ToF
耗时重新核算，不能沿用 IMU 24 秒/epoch，也不能将 base 的耗时当成其余
两个深度架构的实测。base/all_rot 已开始，其他三分支已完成；当前 v1
正常运行，未重跑、未修改训练设置、未提交比赛。

冻结原多传感器方法相同 fold 0 的 1,627 条 OOF 用统一官方指标重新计算
为 **0.8432381633110184**，与日志结果的初步差约 **+0.04525**。原方法
包含 THM，此参考分支只用 IMU + ToF，因此这是方法整体对照，并非单独
ToF encoder 的因果消融。此分数未加历史分配或在线伪标签；最后一轮的
分数按照上游规则保留，不选取中间更高 epoch。仍待真实 checkpoint/OOF
下载核验和另外四折，不能解释为线上分数、五折提升或超过第二名。
证据 `outputs/second_place/evidence/v1_all_completed_log.txt`；JSON 仍保持
`actual_reproduction_scores=null`，将本项放在 `pending_artifact_audit`。


### 首折结果回收与严格核验：2026-10-08 04:25 PDT

v1 成功结束，运行 4 小时 31 分 23 秒；四个 base 分支各训练 50 epoch。
正常浏览器下载的 ZIP 为 74,789,671 字节，SHA256
`e90cdcd1ebbd56496980f0f007ae99745078c55824a2547b466f98907d106b5c`。
同版本 cache/metadata.csv 的下载打开原始 CSV 文档，从浏览器可见 DOM
保存全部 8,151 行及其原始顺序。记录本地保存后的 metadata SHA256；
该值不宣称是云端 CSV 的原始字节哈希。未下载大型训练缓存。

核验脚本验证原始输入字节 SHA256、固定 folds、实际 6,524 train / 1,627
validation sequence 及不重叠 subject、训练集定义的 102 联合类别、四份
50 epoch history 和最后 checkpoint、logits 与排序 CSV 的 ID 对齐、
概率/硬决策/官方指标，以及三个缺失场景的分支路由。均通过。
首次本地回收遇到匿名 pd.Index 导致 reset_index 丢失 sequence_id 列名，
已显式保留列名，新增逆序 ID 回归检查，重新导入独立目录后核验成功。
此修复仅涉及评价代码，云端训练及权重不变；7 项相关测试通过。

| 相同 fold 0 / 1,627 条验证样本 | 冻结方案 | 第二名 base 路由 | 差值 |
| --- | ---: | ---: | ---: |
| 原始输入 | 0.843238 | 0.889177 | +0.045939 |
| 固定一半辅助传感器缺失 | 0.812717 | 0.859828 | +0.047111 |
| IMU-only | 0.775652 | 0.832582 | +0.056930 |

四个独立分支为 imu 0.832582、imu_rot 0.744109、all 0.888491、
all_rot 0.831668。旋转缺失分支的输入条件不同，不直接作为 observed
模型的优劣消融。all 使用 IMU+ToF，冻结多传感器方案另含 THM。

严格因果联合类别 Hungarian 只使用同 subject 已到达的 logits，返回
当前样本且不修改此前结果；三种固定顺序分别报告，不选择最佳顺序。
42 / 142 / 242 的分数：原始输入 0.917111 / 0.918836 / 0.918877；
aux_dropout50 0.892822 / 0.886263 / 0.891081；IMU-only
0.864159 / 0.858959 / 0.867956。各场景和顺序均无 overflow fallback。
在线伪标签尚未评价。以上全部是固定首折开发验证，不是五折均值、
作者十折权重精确复现或线上成绩；不据此替换冻结方案。

紧凑证据 `experiments/results/second_place_base_fold0_v1.json`；权重及
预测存于 `outputs/kaggle_training/imported/second_place_base_fold0_v1_audited`。
下一步按实际 GPU 额度分批完成未训练 folds，复用已完成首折。


### 后续固定 fold 1 已启动：2026-10-08 04:33 PDT

账户菜单明确显示 GPU **22 小时 37 分 available of 30h**；编辑器的
Quota 07:22 表示已用，不能误当剩余。按首折 4h31m23s 的实测估算，
base 剩余四折约 18.09 小时，预计在当前额度内；尚没有三架构五折
均可完成的预算证据。每个后续批次应重新读取实际 available 额度，
并根据实测安排，不启动明知预算不足的任务。

私有 Notebook **CMI Second Place Fixed Fold Reproduction v2**，
scriptVersionId **356387143**，T4 x2、互联网关闭，已确认 Running。
只训练 **base fold 1** 的四个原始分支各 50 epoch，保存最后权重；
不重训已通过核验的 fold 0，不修改 lr/batch/mixup/loss/seed 等参数。
新目录 `base_fold1_v2`，ZIP 显式包含 logit sequence IDs 和小 metadata。
导出器新增可指定结果目录/Notebook文件名的参数，使历史 Notebook
保留；新包源代码和 fixed folds 与本地字节一致，全部代码 cell
编译通过。Notebook SHA256
`ead09b839b33960ea39d4a80b6bda1495edc888da957222961022a9eb1f92694`。

运行链接：
https://www.kaggle.com/code/mingweiwei03/cmi-second-place-fixed-fold-reproduction?scriptVersionId=356387143

没有比赛提交、push 或生产权重替换。cmi 跟进已更新到此版本；
完成后先独立核验，再在真实额度内继续 folds 2/3/4。五折齐全后
才报告五折均值、样本标准差及 8,151 条 OOF；其他架构和在线伪标签
贡献仍未完成。页面运行证据
`outputs/second_place/evidence/kaggle_v2_fold1_running.jpg`。


### 固定 fold 1 结果已回收核验：2026-10-08 09:25 PDT

v2 成功完成，runtime **4h44m13s**；正常浏览器下载包 75,396,076 字节，
SHA256 `d6998d43937aed624d9b1d14587872f3b9d65a4b737903d3bf535e2268212cb4`。
下载事件虽然超时，但 Downloads 实际完整文件存在；复制进独立 recovery_v2，
没有把事件超时当成下载失败或重复启动导出任务。

ZIP 内显式 logit IDs、8,151 行 metadata、四份最后 50 epoch checkpoint/history，
以及三个场景的 logits/probabilities/硬决策/官方指标全部通过核验。
实际 train 6,519 / validation 1,632；固定 subject 分组没有重叠，联合类别
102 个仅来自训练集。ZIP 内 metadata SHA256 与 v1 从浏览器保存的 CSV
完全一致，进一步核实首折元数据的字节和原始行顺序。

| 相同 fold 1 / 1,632 条验证样本 | 冻结方案 | 第二名 base 路由 | 差值 |
| --- | ---: | ---: | ---: |
| 原始输入 | 0.849067 | 0.894506 | +0.045439 |
| 固定一半辅助传感器缺失 | 0.815806 | 0.866117 | +0.050311 |
| IMU-only | 0.785888 | 0.838603 | +0.052715 |

四个独立分支 imu / imu_rot / all / all_rot：
0.838603 / 0.738142 / 0.889365 / 0.836412。
严格因果历史三个顺序 seed42/142/242（各fold实际arrival seed为seed+fold）：
observed 0.934403 / 0.927252 / 0.924043；aux_dropout50
0.901730 / 0.904822 / 0.902734；IMU-only
0.876722 / 0.876831 / 0.870412。没有 overflow 或真验证标签/未来序列参与。
在线伪标签权重更新仍未运行。

两折共 3,259 个独立验证 sequence；仍是部分开发验证，不报五折均值/标准差，
不当成线上或作者十折原权重精确复现。记录
`experiments/results/second_place_base_fold1_v2.json`，导入目录
`outputs/kaggle_training/imported/second_place_base_fold1_v2_audited`。
下一批启动前账户明确 **17h53m available of30h**；剩余三折按 v2实测
估计14.21小时，当前预算预计可覆盖base。三架构完整五折仍无预算保证。


### 固定 fold 2 新版本已接受：2026-10-08 09:27 PDT

私有 Notebook **v3 scriptVersionId356470282** 已接受并显示 Queued。
T4x2、互联网关闭、Pin to original environment (2026-10-02) 保持与 v2 一致。
只训练未完成的 **base fold2** 四个原始分支各50epoch，最后checkpoint，
不重训fold0/1，不改变原训练超参。新输出 `base_fold2_v3`；显式logit IDs
和小metadata包含在ZIP，不需下载多GB缓存。Notebook各code cell编译通过，
嵌入训练脚本、folds、原作者source manifest、输入SHA配置与本地字节一致。
Notebook SHA256
`10b291363d3ef73d281107a762fdb7087d091bc5456409e81346d7a63b6a1880`。

精确运行链接：
https://www.kaggle.com/code/mingweiwei03/cmi-second-place-fixed-fold-reproduction?scriptVersionId=356470282

启动前明确available额度17h53m；cmi跟进已更新到v3。排队尚不能当作
训练完成或成功分数，后续只监控这一次已接受的运行。完成后先核验再
继续未完成fold3/4，所有结果单独保留，本地commit，不push或比赛提交。


2026-10-08 09:52 PDT：v3 已从排队进入 Running，云端再次核验原始 train/demo
SHA256，环境 torch2.11.0+cu128/Tesla T4。base/imu fold2 正常训练，
6,520 train / 1,631 validation、102 联合类别；尚无完整折结果，不报告
中间epoch为最终分数，不重复启动。


### 固定 fold 2 结果回收核验：2026-10-08 14:39 PDT

v3 (356470282) 完成，runtime **4h44m41s**。ZIP 75,836,154 字节，
SHA256 `816e8ff58845391618e7b37b4f40c6c19dc58902edb56111fb8486fb844c1843`。
正常 UI 下载事件超时但 Downloads 实际文件完整，已复制进独立 recovery_v3；
无重复下载任务或重训。metadata SHA 与前两批一致，显式 logit IDs、原始输入字节
SHA、固定 folds、实际 train 6,520 / val 1,631 subjects/IDs、train-only 102
联合类别、四份最后50epoch权重/history、概率/硬决策/官方指标均通过严格核验。
导入 `outputs/kaggle_training/imported/second_place_base_fold2_v3_audited`，
紧凑记录 `experiments/results/second_place_base_fold2_v3.json`。

| 同一 fold2 / 1,631 条验证样本 | 冻结方案 | 第二名 base 路由 | 差值 |
| --- | ---: | ---: | ---: |
| 原始输入 | 0.845631 | 0.874310 | +0.028679 |
| 固定一半辅助传感器缺失 | 0.817661 | 0.847898 | +0.030237 |
| IMU-only | 0.785455 | 0.818403 | +0.032948 |

四分支 imu / imu_rot / all / all_rot 为0.814243 / 0.722499 / 0.869669 / 0.834925。
严格因果联合历史固定到达顺序42/142/242（实际seed+fold），observed
0.904481 / 0.903522 / 0.906080；aux_dropout50
0.874777 / 0.879220 / 0.876835；IMU-only
0.851301 / 0.851085 / 0.849331。overflow均0，无真实验证标签/未来样本参与，
无在线伪标签权重更新，不挑选最佳顺序。

三折共4,890条独立验证sequence，仍非完整五折或线上分数，暂不报告五折均值/标准差。
下一批前账户明确 **GPU13h8m available of30h**；剩余base两折按v3实测
约9.489h，预计在额度内；simple/deep完整五折没有预算保证。
保留已完成三折，不push、不比赛提交、不替换生产权重。


### 固定 fold3 启动：2026-10-08 14:41 PDT

私有v4 **scriptVersionId356543991** 已进入Running，T4x2、互联网关闭，
编辑器Pin to original environment(2026-10-02)与v3一致；日志实际
torch2.11.0+cu128/Tesla T4、固定fold文件SHA已核实。原始输入SHA/cache
仍在启动检查阶段，尚不声称它们已完成。只训练base fold3四分支各50epoch，
bs32与原始超参不变，最后权重，输出`base_fold3_v4`。
Notebook code cells编译及嵌入训练脚本/folds/source/inputhash配置字节核验通过，
SHA256 `0414988b1cdef225d8b790b03fcb436bb3b090c2aa62a589de975f65632a950b`。

https://www.kaggle.com/code/mingweiwei03/cmi-second-place-fixed-fold-reproduction?scriptVersionId=356543991

启动前明确GPU13h8m available，剩余两折按v3实测估计9.489h。
不重训已核验fold0/1/2，无比赛提交/push/生产权重替换。页面证据
`outputs/second_place/evidence/kaggle_v4_fold3_started.jpg`，自动跟进更新到v4。
