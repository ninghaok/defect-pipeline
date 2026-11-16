# 有限额的伪缺陷增强

本模块默认启用，部署前必须配置生成后端。缺少后端配置会在闭环初始化前报错，不会静默退回未增强模式。候选完成阈值校准后直接进入影子；只在配对的真实复核样本上比较 C 和 T₅，具体见 [当前基准](BASELINE.md)。固定测试集不参与阈值选择或模型切换。

## 接入位置和数据隔离

真实数据流 → 复核与原有抽检 → 真实 NG 积累达到训练里程碑 → 固定 train/calibration 划分 → **可选生成、筛选与限额** → YOLO 训练 → 真实校准集调阈值 → 推理冒烟 → 真实影子流比较 → 切换。

- 仅当前候选模型的真实训练 NG 和可信 OK 可作为生成支持。抽检伪 OK 可按原有规则训练 YOLO，但不能训练生成器。历史合成图不能再作为生成支持。
- 初始参考图、校准图、独立测试图不作生成支持。读取测试图的散列仅用于防重复，不使用其标签或分数选图。
- 伪缺陷只附加到当轮 YOLO 训练列表，不写反馈库、真实积累池或持久化分割分配表，不提前触发训练里程碑。
- 原有 NG 里程碑、训练轮数、基础权重和阈值算法均保持原配置。关闭开关时不调用生成器，不改变原训练指纹和数据顺序。

## 初始保守参数（待实验验证，非最优结论）

| 参数 | 默认值 | 含义 |
|---|---:|---|
| min_real_train_ng | 20 | 至少 20 张 ROI 内有效真实训练 NG |
| min_true_ok / max_support_ok | 20 / 60 | 可信 OK 支持下限/上限，最大数须为偶数 |
| max_support_ng | 40 | 按真实缺陷面积分位取最多 40 张支持 NG |
| refresh_new_real_ng | 20 | 相对上次生成至少新增 20 张有效真实训练 NG 才刷新 |
| max_events | 8 | 每类、每次完整运行最多 8 次生成；失败不自动重试 |
| max_train_synthetic | 20 | 每轮 YOLO 最多加入 20 张 |
| max_synthetic_per_real_ng | 0.10 | 伪缺陷不超过有效真实训练 NG 的 10% |
| max_synthetic_fraction | 0.03 | 伪缺陷占当轮训练集不超过 3% |
| max_positive_fraction_increase | 0.02 | 相对加入前，训练 NG 占比最多增加 2 个百分点 |
| area_match_factor | 2 | 合成缺陷面积占比需在对应真实面积分位的 0.5–2 倍以内 |

R 为 ROI 过滤后真实训练 NG 数，O 为当轮全部训练 OK 数，S 为实际加入数。S 同时满足：S≤20、S≤0.1R、S/(R+O+S)≤3%、(R+S)/(R+O+S)−R/(R+O)≤2%。所有上限向下取整后取最小值。比如 R=30、O=100 时最多 3 张；R=100、O=400 时最多 10 张。

生成池和入训池不同。现有 SeaS formal 后端固定训练 800+800 步、最多 3 轮推理，每轮 40 张原始候选，输出 40 张结构合格配对；所以单次最多尝试 120 张，8 次最多 960 张原始候选。这个限额不等于实际入训数。面积筛选允许不足额乃至 0 张，不启动无限补生成。后端无法取得 40 张结构有效配对时会明确报错，不能默默当作成功增强实验。

仅复用最近一轮的生成池，不累计拼接各轮历史池。当前真实支持不再属于训练集时不复用该池。达到总生成上限后仍可对最近池按当轮配额重新筛选。状态先落盘再启动外部进程，重启不会重置次数；失败/中断须检查记录后使用独立重试运行，不自动重试。

## 质量和误报控制

校验请求/输出文件 SHA256、路径范围、图像掩膜同尺寸、白色缺陷二值非空非满掩膜；防止与真实/保留集内容重复。按 ROI 内面积占比匹配真实训练分布，不通过放大掩膜凑面积，不用当前 YOLO 的预测 NG/OK 来挑图。通过结构检查仍不等于外观真实、缺陷与掩膜语义正确；合成掩膜始终记录为伪标注。

保留真实 OK 数量，同时限制合成正例引起的类别比例偏移，用于针对误报风险。此机制不能保证误报必然下降：后续需对相同真实测试集同时核对漏检数、误报数和 micro IoU。建议抽查每个生成事件的最终入训图，重点看正常纹理被标成缺陷、缺陷轮廓错位、过大过亮的生成痕迹。

## 配置与运行

主线只有轻量接口，SeaS 和扩散模型依赖留在独立后端环境。后端需要实现 `augmentation/seas_contract.py` 的请求/输出约定，由部署者单独提供服务脚本及依赖；本仓库不包含 SeaS 生成服务、基础权重或以前实验的微调权重。部署时另行提供已安装 SeaS 的 reuse_root（基础模型、原始代码、环境和工具）。

```bash
python cli/configure_synthetic.py \
  --python /absolute/generation-env/bin/python \
  --service /absolute/generation/service.py \
  --reuse-root /absolute/seas-assets \
  --output /absolute/new-experiment/synthetic.yaml --allow-unreviewed

PIPELINE_SYNTHETIC_CONFIG=/absolute/new-experiment/synthetic.yaml \
PIPELINE_RESULTS_ROOT=/absolute/new-experiment/augmented \
python cli/run_lifecycle.py --category qiumian_xiepai \
  --inbox /absolute/frozen-stream \
  --initialization-manifest /absolute/frozen-stream_manifest.json
```

`--allow-unreviewed` 只表示实验允许结构合格但未经人工逐图验收的伪标注，不是生产验收通过。后端所有同目录 Python 源码均固定散列，源码变化必须重新冻结配置并使用新结果目录。可为四个独立类别进程指定各自配置文件；无需更改主线配置。

未增强对照：显式设置 `PIPELINE_SYNTHETIC_CONFIG` 指向仓库的 `configs/synthetic_baseline.yaml`（`enabled: false`），并指定**新的结果目录**。不能把已用合成图训练出的权重或反馈历史改名当成基线。开启生成时释放预训练模型 GPU 显存，完成后恢复并核对阈值及随机数状态。

## 可溯源输出和大表

- `workspace/augmentation/<类别>/state.json`：策略指纹、累计事件、每轮状态、请求与输出散列。
- `requests/`、`events/`、事件日志：真实支持来源、观测时间边界、生成参数、候选图片与伪掩膜。
- `model_registry/<类别>/milestones/*/training_manifest.json`：每一张实际训练/校准样本的角色、文件、散列、ROI 排除标志及最终 YOLO polygon 文本。数据临时链接删除后依然保留此清单。
- `summary.json`：实际选中数、配额、比例、面积匹配目标、选择/跳过原因和生成事件。
- `cli/export_simulation_table.py`：输出逐批完整仿真指标、实际训练数、合成占比和影子比较；JSON 的 `independent_evaluations` 先列初始预训练，再按里程碑列出每个 YOLO 的明确版本与独立指标。

大表中的“可用于模型训练 OK/NG”是本批结束时累积的可用池，不能当成当轮实际训练量。“该 YOLO 实际训练 OK/NG”是该模型实际物化的数据，启用增强后 NG 包含合成图，右侧分列真实 NG 和伪缺陷。在线指标是当批完整模拟流的事后评测，影子门控只使用真实已复核子集，两者不能替换。最新 YOLO 也可能仍在影子或被拒绝，并不等于当前正式模型。

## 四类对照方案

同一冻结数据流、初始参考/记忆库/校准划分、固定测试集、训练种子、100 轮训练、基础 YOLO 权重与新切换规则；每类建立 baseline 与 augmented 两个独立结果目录。两组通过配置文件显式选择是否增强，主线默认增强。

优先按同一类别/同一流批次比较实际服务指标、累计复核量和最新 YOLO 固定测试指标。切换后会改变复核选择和真实积累，后续实际训练集可能不同，这是完整流程的整体效果，不能解释为单独的合成图因果效果。若需隔离机制效果，另用同一冻结真实训练快照做直接 YOLO 配对对照。

完整流结束但影子证据不足的候选继续标记“影子中”，不能用测试集替代影子验收，也不自动晋升。新规则改变后续流向，历史大表只可做参考，不能修改状态列后冒充新规则重跑。
