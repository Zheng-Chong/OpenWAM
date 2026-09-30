# Project Handoff

更新时间：2026-09-30
当前分支：g1-dex1-finetune（推送到 fork：Zheng-Chong/OpenWAM）
当前目标：G1 mid-train 已完成，在自采桌面数据（G1D）上做后训练；并行准备 AgiBotWorld-Beta 数据（转换 + 规则清洗）

## 项目状态

- 上游仓库 OpenWAM-Official/OpenWAM 的 fork；本分支新增 G1-Dex1 数据读取、统计量计算和离线评估，未改动模型/训练代码。
- 数据：`unitreerobotics/G1_Dex1_*`，DSW 共享盘 `/mnt/data/datasets/Unitree_G1_Dex1`。其中 65 个 LeRobot v3.0、带躯干系末端位姿的任务可用（3353 万帧、30 fps，`recomputed_ee_valid` 全为 1，无 NaN）；v2.1 和纯关节格式的任务在发现阶段跳过。
- 动作契约（α 80 维）：每只手臂 `[xyz3（躯干系）, rot6d6, open1]`，左臂在前，映射 `["0-9", "34-43"]`；欧拉角为外旋 XYZ（R = Rz·Ry·Rx，已用 state_torso 复合验证误差为 0）；夹爪 `clip(gripper_pos/4.5,0,1)*2-1`（4.5 = 张开）。
- 所有任务共用一份统计量，只在训练集上计算（action + state 合并），rot6d 和夹爪维度固定为恒等映射：`/mnt/data/chongzheng/openwam_g1/g1_dex1_normalization_stats.npy`。
- 划分：`episode_index % 50 == 0` 为验证集（约 2%）。
- 基础模型：`/mnt/data/models/OpenWAM-Alpha-Pretrain-Foundation-Model`（step 154000，24.8 GB）。
- G1 关键约束：只用 v3.0 躯干系末端位姿任务；统计量 min-max；训练输出必须放本地盘（ossfs 不支持 safetensors 写入）；训练端可开 `training.prompt_embed_cache_size` 缓存 umT5 编码（每步约省 1.2 s）。
- G1 mid-train 已完成（2026-09-29）：dsw-1 `/root/openwam_g1/runs/2026-09-27_11-35-37/checkpoint_step_20000.safetensors`，6000–20000 步副本在 `/mnt/data/chongzheng/openwam_g1/ckpts/stepN/`。验证集（1300 窗口）位置误差 77.1 mm（基座）→ 27.1 mm（20000 步），旋转 15.3° → 7.8°，夹爪准确率 88% → 97%；各 checkpoint 结果在 dsw-share1 `/root/eval/stepN/summary.json`。
- G1D 自采数据（`/mnt/data/datasets/G1D-自采数据`）已转成 torso-EEF 格式：`/mnt/data/datasets/G1D-自采数据-eef/`（6 个 bucket，948 条 episode，54.5 万帧），读取器 `g1d_self`（配置 `configs/dataloader/g1d_self.yaml`）。
- G1D 后训练 2026-09-29 11:02 在 dsw-1 启动：mid-train step 20000 初始化，8 卡 × 每卡 16 × 梯度累积 2 = 全局 256，lr 1e-4（视频 / 动作统一），cosine、5% warmup，prompt 缓存开，`max_steps=20000` / `save_steps=2000`（均按 micro 步计，即 10000 优化器步、约 5.2 epoch），约 5.2 s/micro 步，预计 28.5 小时。输出 `/root/openwam_g1/runs_post/2026-09-29_11-01-56/`，日志 `/root/openwam_g1_logs/posttrain.log`，启动脚本 `/root/openwam_g1/posttrain.sh`。
- **2026-09-29 20:00 dsw-1 后训练在 micro 6483 步崩溃**（dsw-1 的 ossfs `/mnt/data` 掉线，读视频失败）。2026-09-30 00:25 在 dsw-8 从 step 6000 权重续训：`training.start_step=6000`（新增配置，计步和 LR 调度快进到原进度，首步 lr=8.40e-5 与原训练一致；优化器状态未保存，无法恢复），`save_full_states_for_resume=true`。数据和权重全部先复制到 dsw-8 本地 `/root/g1d_data/`（`eef/` 的 `video_path` 改指本地 `raw/`）。启动脚本 `/root/openwam_g1/posttrain2.sh`，输出 `/root/openwam_g1/runs_post/2026-09-30_00-25-01/`，日志 `/root/openwam_g1_logs/posttrain2.log`，dsw-8 `/root/ckpt_sync_post.sh` 复制新 checkpoint 到 OSS。dsw-8 没有 rsync / cpio，用 tar 管道。
- 后训练自动化：dsw-1 `/root/ckpt_sync_post.sh` 复制到 `/mnt/data/chongzheng/openwam_g1/ckpts_post/stepN/`；dsw-share1 `/root/eval_watch_post.sh` 先评估 step 0（mid-train 权重）再评估每个新 checkpoint（每任务 100 个验证窗口），结果 `/root/eval_post/stepN/summary.json`。
- AgiBotWorld-Beta：原始数据 `/mnt/data/datasets/agibot_world_beta`（tar，仍在提取到 `agibot_world_beta_extracted`）；LeRobot v3 转换产物在 dsw-2 本地盘 `/root/AgiBotWorld-Beta-lerobotv3`。

## 最近任务

### 目标
- 把 AgiBotWorld-Beta 原始数据转成 `agibotworld` 读取器要的 LeRobot v3，并做规则式数据清洗（长度、数值、跳变、静止、指令、视频）。

### 已完成
- `openwam/dataloader/utils/agibotworld_convert.py`：原始 h5 + task_info + mp4 → 每任务一个 bucket；每条 episode 一个 parquet，视频软链到原始 AV1 mp4（不重编码）；`episode_index` = 原始 episode_id；`meta/info.json` 最后写，作为完成标记，重跑默认跳过已完成 bucket（`--overwrite` 重转）。
- `openwam/dataloader/utils/episode_quality.py`：逐 episode 计算指标 → `quality.parquet` + `summary.json`；`--apply` 才把标出的 episode 合并进各 bucket 的 `meta/excluded_episodes.json`（复用 `exclusion_io` 的锁 + 原子写，附 `episode_quality` 原因）。视频检查只解码 5 个关键帧包（AV1 需 flush），约 0.1 s/条。
- dsw-2 上全量转换：已提取的 199 个任务中 166 个成功，127,929 条 episode、2191 小时，产物 `/root/AgiBotWorld-Beta-lerobotv3`（41 GB，本地盘）。18 个任务无 h5、15 个无 task_info；46 个任务提取不完整（比 task_info 少 7298 条）；2 条视频损坏被跳过；灵巧手任务尚未提取，0 个。
- 全量扫描（45 分钟，64 进程）：标出 2882 条（2.3%，56.8 小时），`pos_jump` 2861、`rot_jump` 32，其余规则（长度、NaN、静止、指令、黑屏 / 冻结 / 解码失败）全部 0 命中。结果：`/root/agibot_quality/`。**尚未 `--apply`**。

### 关键决策
- 字段语义按官方 README 核对并实测：四元数 xyzw；末端是 flange 位姿（m，底盘系）；夹爪 action 0=开 / 1=合，state 单位 mm（35–125）→ 除以 1000 对上读取器的 0.035/0.125 m 标定（已用 action 与 state 的对应关系验证方向）；底盘 action 速度 `[vx, yaw]` → `[vx, 0, yaw]`，state 无来源写 0。
- action 位姿 = 下一帧 state（读取器约定已做 next-state 重标）；夹爪 action 用原始 action 通道。
- 每帧 prompt 用 `label_info.action_config` 的子任务文本，未覆盖帧用 `task_name`。
- 产物放本地盘：ossfs 不支持软链。其他节点训练前 `rsync -a` 复制（软链指向共享的 `/mnt/data`，复制后仍有效）。
- 原始数据没有 `segment_flag`，读取器用完整 episode（首尾静止帧未裁）。
- 跳变阈值由 5 cm/帧放宽到 10 cm/帧：在任务 373 上，5 cm 标出的都是正常分布尾部的快速动作。
- 位置跳变的成因：左右臂同一帧沿 z 平移相同距离、x/y 不变，推断是升降腰高度信号跳变（707/764 为单帧尖峰，725 为 0.5 m 阶跃）。725（Scan security check）和 748 两个任务 100% 被标出，整任务剔除会损失场景，所以先不 apply，等用户决定。

### 2026-09-29 补数据与重转
- 本地原始数据当初来自 ModelScope 镜像 `agibot_world/agibot_world_beta`（1121 个 tar），它比 HF 少 112 个 tar，另有 4 个是 136 字节的占位文件、3 个和 HF 版本不同，所以本地缺数据。缺的部分只能从 HF 补：用 `/mnt/data/chongzheng/agibot_beta_fetch/agibot_beta_fetch.py`（不进仓库），token 在 dsw-2 的 `~/.cache/huggingface/token`，走 hf-mirror。
- 第一轮提取发现很多已下载的 tar 当初没提取完，已补齐（这轮顺带提出了 5 路鱼眼视频，多占空间但无害）。
- 灵巧手机器人只有鱼眼腕部相机（960×768）：转换脚本改为腕部相机按候选列表取第一个存在的文件；提取白名单也包含鱼眼腕部。
- 转换报告改为按任务合并（之前单任务重跑会覆盖全量报告）。
- 重转后：213 个 bucket、156,304 条、2578.5 小时，其中灵巧手 19 个（5851 条，读取器实测每只手 9 维位姿 + 6 维手指被监督）。仍有 60 个任务缺 10,624 条，等 HF 下载完成。
- 重新扫描（`/root/agibot_quality_v2/`，未 apply）：标出 2910 条（1.9%），`pos_jump` 2873、`rot_jump` 57；灵巧手只标出 18 条。

### 2026-09-29 删除不合格 episode 并写入 OSS
- 用户决定直接删除：`episode_quality --quality <已有 quality.parquet> --delete` 物理删除 2910 条（data 文件、视频链接、episodes 行，更新 info 总数；原因记在各 bucket 的 `meta/deleted_episodes.json`）。725（Scan security check，952 条）和 748（104 条）被删空，整个 bucket 移除，记录在根目录 `deleted_buckets.json`。删除前的 meta 备份：dsw-2 `/root/agibot_lerobotv3_meta_backup_20260929.tar`。
- 删除后：211 个 bucket、153,394 条、2521.4 小时。
- **已复制到 OSS 并核对**：`/mnt/data/datasets/AgiBot/AgiBotWorld-Beta-lerobotv3`，8.48 TB（视频 8.44 TB 实体文件，460,182 个 = 3 × 153,394），读取器取样正常。这是正式版本；dsw-2 本地 `/root/AgiBotWorld-Beta-lerobotv3` 只是工作副本。
- 复制注意：ossfs 不支持 `ftruncate`，不能用 `rsync --inplace`（errno 95）；用 `rsync -rL --size-only`（脚本 dsw-2 `/root/owam_conv2/copy_to_oss.sh`）。
- 原始数据路径以 `/mnt/data/datasets/AgiBot/agibot_world_beta` 为准（2026-09-29 由他人迁移）。补下载已改到新路径（搬迁期间写到旧路径的 30 个 tar 已挪过去）；提取输出 `agibot_world_beta_extracted` 仍在旧位置 `/mnt/data/datasets/`（上百万个文件，ossfs 目录改名要逐个对象改，未迁移）。

### 涉及范围
- `openwam/dataloader/utils/agibotworld_convert.py`、`openwam/dataloader/utils/episode_quality.py`（新增）。
- `tests/dataloader/test_agibotworld_convert.py`（原始 → bucket → `AgiBotWorldDataset` 往返）、`tests/dataloader/test_episode_quality.py`。
- 读取器、统计量代码未改。

### 验证
- DSW（profile 标签 dsw-h20x8，系统 Python）隔离目录：`python3 -m pytest -q tests/dataloader/test_agibotworld_convert.py tests/dataloader/test_agibotworld.py tests/dataloader/test_episode_quality.py` → 45 passed。
- 任务 373 真实读取：AV1 解码正常，三视角拼图正确，夹爪闭合为 0，底盘两维被监督（掩码 22 维）。
- 本机没有 h5py，转换测试在本机会跳过。

### G1D 自采数据后训练（2026-09-29）
- 排查记录：第一次后训练 debug 在第 6 步 NCCL all-reduce 超时。原因是 rank 4 读到视频被截断的 episode，`_safe_get` 重试用完后抛错；非 0 号 rank 的调用栈又被 `scripts/train.py` 吞掉（`builtins.print` 在非主 rank 是空函数，`traceback.print_exc` 走的是 print），已改成 `sys.stderr.write(format_exc())`。Hydra 命令行里的中文路径要加引号：`"dataloader.dataset_dir='/mnt/.../G1D-自采数据-eef'"`。
- 目标：用 mid-train 20000 步权重，在自采桌面任务上混合后训练，提高这些任务的成功率。用户决定：不跑 α 基座初始化的对照；所有自采任务混在一起；只要桌面操作。
- `openwam/dataloader/utils/g1d_self_convert.py`：关节角 → URDF 正运动学（`torso_link → <side>_dex1_base_link`，再沿局部 x 平移 0.0635 m）→ 与官方完全相同的 `*_ee_pose_gripper_torso` 列。在官方 ZipUp / Arrange_Flowers 上核对：位置误差 0.000 mm、旋转误差 0.000°。URDF 放在 `openwam/dataloader/assets/unitree_g1_dex1.urdf`。
- 夹爪：自采 Dex1 张开约 5.4（每个 episode 开头 5.37–5.38），按 4.5/5.4 缩放到官方刻度，之后沿用 `clip(g/4.5,0,1)*2-1`。
- 过滤：`is_bad`、底盘 XY 位移 > 1 cm、偏航 > 1°、升降变化 > 1e-3、底盘速度指令 > 0.02、数据长度超过任一相机 mp4 实际帧数的整条 episode 丢弃；实际丢了 PourBeans 1 条（指令峰值 0.025）和 PourBeansPlus 2 条（episode 4、50 的视频文件被截断，元数据的 to_timestamp 不可信，要读 mp4 本身）。PourBeansEps380 用用户裁掉底盘移动前缀的 `_desk` 版本，原版不转换。
- 数据格式：A 批（PickKettle 16 维、PourBeans / PourBeansPlus 23 维 FullObs）只有关节角；B 批（capybara、pick_3objects、pick_bottle，可移动升降底盘）有 `arm_pose`，但位置参考点和官方差约 70 mm，统一改用关节角 + 正运动学。B 批关节顺序已用 `arm_pose` 的旋转核对（0.000°）。
- 指令：没有改写的任务补了 5 条英文改写（`@` 分隔）；PourBeans380 的英文改写原本只在 episodes 表的 `tasks` 列，转换时写进 `tasks.parquet`。
- 读取器 `G1DSelfDataset`（`g1_dex1.py`）：只换相机名（`cam_left_high` / `cam_left_wrist` / `cam_right_wrist`），视频帧偏移改为 `round(from_timestamp × fps)`，因为视频原地共享、episode 被裁剪 / 过滤后按长度累加会错位。未裁剪的 Kettle 上两种算法结果一致。
- 统计量沿用 mid-train 的文件，保证动作归一化和 checkpoint 一致；抽样 60 个窗口，没有值超出 [-1, 1]。验证集 `val_every: 10`。
- 验证：dsw-share1 `pytest tests/dataloader/test_g1d_self_convert.py tests/dataloader/test_g1_dex1.py` → 6 passed（含与官方位姿逐帧对比）；读取器训练 48.8 万 / 验证 5.6 万窗口，视频解码和三视角拼图目检正确。

## 其他数据集（2026-09-30）
- 通用约定：新转换的数据集夹爪统一 `[0,1]`，0=闭合 1=张开；原始量程写进 `info.json`。旋转跳变阈值 20°/步 不变，保留 `too_long`。
- **Galaxea**（`galaxea_convert.py`）：已完成。只保留桌面操作（`--max-chassis-cmd-frac 0.01 --max-torso-range 0.05`）并删除官方/录制质检不合格和规则命中 → 6,147 条 / 93.6 h / 480 GB，OSS `/mnt/data/datasets/Galaxea-lerobotv3`。本地副本已删。删除前 meta 备份：dsw-3 `/root/galaxea_meta_backup_20260930.tar`。
- **Hy-Embodied**（`hy_embodied_convert.py`）：table_000 试跑通过；全量 22 张表转换 + 扫描在 dsw-4 跑（`/root/owam_hy/hy_full.sh`，日志 `/root/openwam_g1_logs/hy_full.log`，产物 `/root/Hy-Embodied-lerobotv3`）。完成后：审查扫描 → 删除 → 复制到 OSS → Notion。
- **lingbot-GM-100**：subagent 在 dsw-4 做（代码在它的 worktree，未合并）。R1 Pro 已转（16,772 条 / 152.4 h，`/root/lingbot-lerobotv3`）；AgiBot G1、AgileX 只有关节角，按用户决定做 FK（进行中）。
- 基础设施：dsw-2 ossfs 挂载断开（需平台重挂），dsw-1、dsw-5 SSH 被拒（dsw-1 上有 G1 mid-train，需确认）。AgiBot 第二轮提取在 dsw-3 重跑（日志 `agibot_extract3.log`），完成后对 60 个不完整任务重转并补扫描、删除、同步 OSS。

## 剩余事项与风险

- AgiBot：HF 补下载进行中（dsw-2，日志 `/root/openwam_g1_logs/agibot_download.log`），完成后 `then_extract.sh` 自动跑第二轮提取（日志 `agibot_extract2.log`）；之后对不完整任务 `--overwrite` 重转并重新扫描。下载完成后提醒用户作废 HF token（曾在对话中明文出现）。
- AgiBot：（已决定删除，已执行）原清洗决策记录：是否 `--apply` 当前清洗结果（2.3%，含 725 / 748 整任务）；可选改为修复单帧 z 尖峰而不是丢弃。apply 后必须重新生成 `stats_g2a.json`（读取器会校验排除后的样本集合）。
- AgiBot：提取完成后对不完整任务（conversion_report 中 `episodes + skipped < task_info_episodes`）加 `--overwrite` 重转；灵巧手任务（`_DEX_BUCKET_IDS`）的转换分支未经真实数据验证。
- AgiBot：还没生成 `stats_g2a.json`，也没用它训练过；首尾静止帧未裁（无 `segment_flag`）；基于 VLM 的指令一致性 / 成功判断未做。

- G1 mid-train（2026-09-27 至 09-29，dsw-1，8 卡，每卡 batch 24，20000 步，video_lr=3e-5 / action_lr=1e-4）已完成；step 2000 / 4000 被滚动删除，没有验证点。`keep_last_k_ckpts` 默认已改为 null（全部保留）。
- G1D 后训练：部署客户端（EEF → IK → 关节）仍未实现，真机成功率要等它做完才能测。
- 只在 G1 数据上训练，其他本体的能力会退化，这是有意为之。
- min-max 统计量受位置离群值影响：大部分数据只占 [-1,1] 中间约一半区间。如果精度不理想，可以考虑清洗离群轨迹。
- `wandb` 在 DSW 上未登录，需设置 `WANDB_MODE=offline`。
- 部署客户端（EEF → IK → 关节）尚未实现，本阶段只做离线评估。

## 下一会话

0. AgiBot 清洗：看 `/root/agibot_quality/summary.json`（dsw-2），用户确认后：`python3 -m openwam.dataloader.utils.episode_quality --root /root/AgiBotWorld-Beta-lerobotv3 --out /root/agibot_quality --video --workers 64 --apply`，再跑 `agibotworld_stats_computation --dataset_dir /root/AgiBotWorld-Beta-lerobotv3 --segment-max-trim-ratio 0.7`。
1. 检查 dsw-1 上的 mid-train 进度：`tail -c 2000 /root/openwam_g1_logs/midtrain.log | tr '\r' '\n' | tail -2`；可以在 dsw-2 到 dsw-7 上对中间 checkpoint 做离线评估（checkpoint 在 dsw-1 本地盘，需要先复制过去）。
2. 训练完成后：
   ```
   for i in 0..7: CUDA_VISIBLE_DEVICES=$i python3 scripts/eval_offline.py --ckpt-dir <run> --out <dir>/shard$i.jsonl --shard $i --num-shards 8
   python3 scripts/eval_offline.py --summarize <dir>/shard*.jsonl --summary-out <dir>/summary.json
   ```

## 最近历史

- 2026-09-30：新增 Hy-Embodied、Galaxea 转换和 official/bag/body_motion 等清洗规则；Galaxea 桌面子集写入 OSS；Hy 全量转换、lingbot FK 进行中。

- 2026-09-29：G1D 自采数据转成 torso-EEF（URDF 正运动学，官方数据上误差 0），新增 `g1d_self` 读取器；准备后训练。
- 2026-09-29：AgiBot LeRobot v3 复制到 OSS（8.48 TB）并核对；原始数据路径迁到 `AgiBot/`，补下载随之迁移。
- 2026-09-29：`episode_quality` 新增 `--quality`（复用结果）和 `--delete`（物理删除，空 bucket 整体移除）；删除 2910 条不合格 episode，开始复制到 OSS。
- 2026-09-29：从 HF 补下载 AgiBot 缺失 tar 并补提取；转换支持灵巧手鱼眼腕部相机，报告按任务合并；重转 213 个任务并重新扫描。
- 2026-09-28：新增 AgiBotWorld-Beta 原始 → LeRobot v3 转换和规则式 episode 质量扫描；全量转换 12.8 万条，扫描标出 2.3%（未 apply）。

- 2026-09-27：分析训练速度（瓶颈在计算，不在数据），新增训练端 prompt 编码缓存。
- 2026-09-27：启动 G1 mid-train（全参，视频 / 动作分开设学习率，20000 步）。
- 2026-09-27：新增 G1-Dex1 多任务读取器、共享统计量、离线评估脚本；DSW 上 debug 微调和离线评估均已跑通。
