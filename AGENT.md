# AGENT.md

给在本仓库工作的编码 agent 看的说明。项目背景看 `README.md`，当前进度看 `HANDOFF.md`，服务器看 `docs/h20.md`。

## 仓库状态

- 这是上游 `OpenWAM-Official/OpenWAM` 的 fork，工作分支 `g1-dex1-finetune`，推送到远端 `fork`（`Zheng-Chong/OpenWAM`）。
  **不要推 `origin`（上游）**，提交和推送都要用户同意。
- 当前任务：用 Unitree G1-Dex1 多任务数据对 OpenWAM-α 预训练基座做全参 mid-train，得到专门的 G1 基座。
  现阶段只做离线（开环）评估，没有部署客户端。
- 每次会话结束前更新 `HANDOFF.md`（项目状态 / 最近任务 / 剩余事项与风险 / 下一会话 / 最近历史）。

## 代码结构（本分支相关部分）

| 路径 | 作用 |
|---|---|
| `openwam/dataloader/g1_dex1.py` | G1-Dex1 读取器：EEF20 动作契约、训练 / 验证划分、多任务聚合 |
| `openwam/dataloader/utils/stats_computation/g1_dex1_stats_computation.py` | 共享归一化统计量 CLI |
| `configs/dataloader/g1_dex1.yaml` | 数据配置；`dataset_dir` 和统计量路径是占位符，启动时覆盖 |
| `scripts/train.sh` / `scripts/train.py` | 训练入口（torchrun + DeepSpeed ZeRO-2 + Hydra） |
| `scripts/eval_offline.py` | 离线评估：位置 mm、旋转度、夹爪 L1 和准确率，按预测步段统计，和"保持不动"基线对比 |
| `openwam/model/video_backbone/wan/encode.py` | `encode_text_cached`：训练时缓存冻结的 umT5 输出 |

## 不能改错的约定

- **动作契约**：每只手臂 `[xyz3（躯干系）, rot6d6, open1]`，左臂在前，映射到 α 80 维的 `["0-9", "34-43"]`。
- **欧拉角**是外旋 XYZ（`R = Rz @ Ry @ Rx`），用位姿复合实测验证过，不要改成别的约定。
- **夹爪**：原始 `gripper_pos` 0 = 闭合、约 4.5 = 张开，转换为 `clip(g/4.5,0,1)*2-1`（−1 闭合 / +1 张开）。
- **统计量**只用训练集计算，rot6d 和夹爪维度固定为恒等映射；读取器会校验统计文件里的 `STATS_CONTRACT`，不匹配直接报错。
- **验证集**是 `episode_index % 50 == 0`，训练永远不能读到。
- `finetune_ckpt_path`（新 run，只加载权重）和 `resume_ckpt_path`（续训，需要 `save_full_states_for_resume: true`）互斥。

## 测试

```bash
python3 -m pytest -q tests/dataloader/test_g1_dex1.py tests/test_eval_offline.py tests/test_prompt_embed_cache.py tests/test_action_normalization.py
```

纯数值测试本机就能跑。真实数据测试需要 `G1_DEX1_ROOT` 和 `G1_DEX1_STATS`，只能在 DSW 上跑，否则自动跳过。
部分已有测试需要 Cosmos 权重，在 DSW 上也会跳过，属正常现象。

## 服务器

详见 `docs/h20.md`。要点：

- 只用 SSH 别名（`dsw-1` 等）。**不要在仓库、日志、HANDOFF、Notion 里写主机、IP、端口、用户、密钥路径。**
- 不在服务器上做 git 操作，代码按已跟踪文件清单 rsync 到隔离目录。
- 训练输出放本地盘 `/root/...`（ossfs 上存 safetensors 会失败）；checkpoint 另外复制一份到 `/mnt/data/chongzheng/openwam_g1/ckpts/`。
- 训练/评估读的数据（parquet、视频）和 checkpoint 权重不能直接从 OSS（`/mnt/data`，ossfs）读，先复制到本机本地盘（如 `/root/g1d_data/`、`/root/ckpts/`），`info.json` 里 `video_path` 等绝对路径同步改成本地路径（2026-09-29 G1D 后训练在 dsw-1 跑到 micro 第 6483 步时 ossfs 掉线 `传输端点尚未连接`，读视频失败导致训练退出）。OSS 只做跨机持久副本（顺序 `cp`/`rsync`），每台机器首次使用时复制一次。
- 训练日志（`*.log`、`debug_loss_history.csv`、wandb offline 目录）也要定期复制到 OSS（如 `/mnt/data/chongzheng/openwam_g1/logs/`），和 checkpoint 一样放进同步脚本：DSW 实例可能被重建，`/root` 会整个清空（2026-09-30 dsw-1 重建后，mid-train 日志和第一次后训练前 6483 步的日志都丢了）。
- 长任务用 `setsid nohup ... & disown` 起。
- 不碰别人的进程；不重启正在跑的训练，除非用户明确同意。

## 常用命令

```bash
# 训练进度（dsw-1）
ssh dsw-1 'tail -c 2000 /root/openwam_g1_logs/midtrain.log | tr "\r" "\n" | grep Training | tail -1'

# 离线评估（8 卡分片 + 汇总）
for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i python3 scripts/eval_offline.py \
  --ckpt-dir <ckpt_dir> --out <out>/shard$i.jsonl --shard $i --num-shards 8 & done; wait
python3 scripts/eval_offline.py --summarize <out>/shard*.jsonl --summary-out <out>/summary.json
```

`<ckpt_dir>` 下需要 `config.yaml`、`normalization_stats.npy`、`tokenizer/` 和一个 `checkpoint_step_*.safetensors`；有多个时自动取最新的。
