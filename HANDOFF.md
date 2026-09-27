# Project Handoff

更新时间：2026-09-27
当前分支：g1-dex1-finetune（推送到 fork：Zheng-Chong/OpenWAM）
当前目标：在 Unitree G1-Dex1 多任务数据上微调 OpenWAM-α，先做离线（开环）评估

## 项目状态

- 上游仓库 OpenWAM-Official/OpenWAM 的 fork；本分支新增 G1-Dex1 数据读取、统计量计算和离线评估，未改动模型/训练代码。
- 数据：`unitreerobotics/G1_Dex1_*`，DSW 共享盘 `/mnt/data/datasets/Unitree_G1_Dex1`。其中 65 个 LeRobot v3.0、带躯干系末端位姿的任务可用（3353 万帧、30 fps，`recomputed_ee_valid` 全为 1，无 NaN）；v2.1 和纯关节格式的任务在发现阶段跳过。
- 动作契约（α 80 维）：每只手臂 `[xyz3（躯干系）, rot6d6, open1]`，左臂在前，映射 `["0-9", "34-43"]`；欧拉角为外旋 XYZ（R = Rz·Ry·Rx，已用 state_torso 复合验证误差为 0）；夹爪 `clip(gripper_pos/4.5,0,1)*2-1`（4.5 = 张开）。
- 所有任务共用一份统计量，只在训练集上计算（action + state 合并），rot6d 和夹爪维度固定为恒等映射：`/mnt/data/chongzheng/openwam_g1/g1_dex1_normalization_stats.npy`。
- 划分：`episode_index % 50 == 0` 为验证集（约 2%）。
- 基础模型：`/mnt/data/models/OpenWAM-Alpha-Pretrain-Foundation-Model`（step 154000，24.8 GB）。

## 最近任务

### 目标
- 为 G1-Dex1 多任务微调打通数据 → 训练 → 离线评估的完整链路。

### 已完成
- 数据读取器 `g1_dex1`（单任务读取 + 根目录多任务聚合，显式列出 tasks 时如有任务加载失败会直接报错）。
- 共享统计量的计算 CLI，已在 DSW 上生成统计文件。
- `scripts/eval_offline.py`：在验证集窗口上开环推理，按物理单位计算指标（位置 mm、旋转角度、夹爪误差和开合准确率，按 horizon 分段），并与"保持当前位姿"基线对比；支持多卡分片和结果汇总。
- DSW 8×H20 上跑通 debug 微调（20 步，每卡 batch 24，约 8.9 s/step，显存充足），并在其 checkpoint 上跑通离线评估（单卡约 0.4 s/块，不开编译）。

### 关键决策
- 只用 v3.0 躯干系末端位姿的任务：腰部会动，而且 Unitree 的 IK 也在躯干系下求解。
- 统计量按 α 规定用 min-max；位置维度存在离群值（min/max 明显宽于 q01/q99），暂不改，记为风险。
- 末端无效帧的处理：窗口内有无效帧就抛异常，由 `_safe_get` 跳到下一个样本。当前数据中没有无效帧，所以不会触发。
- 训练输出必须放本地盘（`/root/...`）：ossfs 不支持 safetensors 的写入方式（os error 95）。

### 涉及范围
- `openwam/dataloader/g1_dex1.py`：读取器、转换函数、多任务聚合。
- `openwam/dataloader/registry.py`：注册 `g1_dex1`。
- `openwam/dataloader/utils/stats_computation/g1_dex1_stats_computation.py`：统计量计算 CLI。
- `configs/dataloader/g1_dex1.yaml`：数据配置（`dataset_dir` 和统计量路径为占位符，启动时通过命令行覆盖）。
- `scripts/eval_offline.py`：离线评估。
- `tests/dataloader/test_g1_dex1.py`、`tests/test_eval_offline.py`：测试。

### 验证
- DSW（profile 标签 dsw-h20x8，系统 Python，env=none）隔离目录中运行：`G1_DEX1_ROOT=... G1_DEX1_STATS=... python3 -m pytest -q tests/dataloader/test_g1_dex1.py tests/test_eval_offline.py tests/test_action_normalization.py` → 43 passed。
- 注意：技能自带的 `remote_validate.py` 会因为 git 子模块 `third_party/cosmos-predict2.5` 拒绝执行，因此改为手动 rsync 已跟踪文件到唯一临时目录，运行后删除。
- 8 卡 debug 训练 20 步成功，保存 checkpoint 正常；在该 checkpoint 上离线评估 5 个窗口正常。

## 剩余事项与风险

- 正式训练尚未启动（需要用户确认步数和资源）。
- min-max 统计量受位置离群值影响：大部分数据只占 [-1,1] 中间约一半区间。如果精度不理想，可以考虑清洗离群轨迹。
- `wandb` 在 DSW 上未登录，需设置 `WANDB_MODE=offline`。
- 部署客户端（EEF → IK → 关节）尚未实现，本阶段只做离线评估。

## 下一会话

1. 检查 DSW 上的正式训练是否在运行：`/root/openwam_g1_logs/`、`/root/openwam_g1/runs/`。
2. 训练完成后：
   ```
   for i in 0..7: CUDA_VISIBLE_DEVICES=$i python3 scripts/eval_offline.py --ckpt-dir <run> --out <dir>/shard$i.jsonl --shard $i --num-shards 8
   python3 scripts/eval_offline.py --summarize <dir>/shard*.jsonl --summary-out <dir>/summary.json
   ```

## 最近历史

- 2026-09-27：新增 G1-Dex1 多任务读取器、共享统计量、离线评估脚本；DSW 上 debug 微调和离线评估均已跑通。
