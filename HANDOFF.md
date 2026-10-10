# Project Handoff

更新时间：2026-10-10
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

## G1D 自采后训练部署（2026-09-30）
- `scripts/serve_g1d.py`：对接机器人端现有的 openpi 协议客户端（msgpack，`observation/state` 关节 16 维 + 三路图像 → `actions [30,16]` 绝对关节角，夹爪原始 Dex1 单位）。服务端做 FK（关节→EEF20 本体感知）和阻尼最小二乘 IK（EEF20 动作块→关节，7 维臂第 7 自由度软拉向当前关节，关节限位来自 URDF）。`--self-test` 验证 FK→IK 往返：位置 <0.3 mm、旋转 <0.02°。`G1ArmFK` 新增 `transform`/`limits`，`pose` 复用。
- **当前部署（2026-10-08）：step 20000（后训练完成）在 dsw-8**：权重 `/root/ckpts_g1d/step20000`（软链到 `runs_post/2026-09-30_00-25-01`），代码 `/root/openwam_deploy`，启动脚本 `/root/g1d_serve.sh <名> <端口> <GPU> "<指令>"`，日志 `/root/g1d_serve_logs/`，探针 `/root/probe_g1d.py <端口>`。GPU0–4 依次：8001 capybara、8002 bottle、8003 marker、8004 plush→basket、8005 倒豆子-Plus（"Pour the beans."）。每次请求约 0.3 s（8 卡全空的 H20，`--compile`，启动预热约 15 s 且 websocket 关 ping）。dsw-6 上 step 16000 的服务已停（权重/代码仍在 `/root/ckpts_g1d`、`/root/openwam_deploy`）。
- 服务实现要点：`scripts/serve_g1d.py` 对接机器人端 openpi 协议（`observation/state` 关节 16 维 + 三路图像 + 可选 `prompt` → `actions [30,16]` 绝对关节角，夹爪原始 Dex1 单位）；服务端做 FK（关节→EEF20 本体感知）和阻尼最小二乘 IK（7 维臂第 7 自由度软拉向当前关节，限位取自 URDF）；IK 30 步×2 臂 0.04 s，自测位置 <0.1 mm / 旋转 <0.02°；compile 离线评估精度不变（模型延迟 0.71→0.58 s，dsw-6 36 窗口）。
- 网关公网转发已开（用户同意）：10076→8001 capybara、10077→8002 bottle、10078→8003 marker、10079→8004 plush→basket、10080→8005 倒豆子-Plus，2026-10-08 起指向 dsw-8（之前指向 dsw-6）。网关上 autossh 常驻，日志 `tunnel_1007x.log`。
- 机器人端客户端（`scripts/unitree_client/`，仓库里是脱敏副本，机器人上 `/home/unitree/client/jepa_g1d_client/` 是带真实地址的原件，旧文件备份 `.bak_pre_*`）：`MODEL_REGISTRY` 加了 OpenWAM 条目；新增启动时选 prompt（`PROMPTS` 11 条训练指令 + 自定义，仅对 `policy` 以 `OpenWAM` 开头的服务弹出；`--prompt` / 环境变量 `PROMPT` 跳过选单），`JepaWamClient.prompt` 随每次请求发送。仓库副本用环境变量 `G1D_GATEWAY_HOST` 代替网关地址。运行要用 `unitree_lerobot` conda 环境（有 msgpack）。
- **未完成**：真机闭环、IK 关节连续性/限位表现未测；机器人 `ssh unitree` 在 2026-10-08 换权重时连不上（网关 10088 关闭连接），机器人上的 `MODEL_REGISTRY` 标签仍写 step16000（仓库副本已改为 step20000），机器人在线后同步标签即可，功能不受影响。

## 其他数据集（2026-09-30）
- 通用约定：新转换的数据集夹爪统一 `[0,1]`，0=闭合 1=张开；原始量程写进 `info.json`。旋转跳变阈值 20°/步 不变，保留 `too_long`。
- **Galaxea**（`galaxea_convert.py`）：已完成。只保留桌面操作（`--max-chassis-cmd-frac 0.01 --max-torso-range 0.05`）并删除官方/录制质检不合格和规则命中 → 6,147 条 / 93.6 h / 480 GB，OSS `/mnt/data/datasets/Galaxea-lerobotv3`。本地副本已删。删除前 meta 备份：dsw-3 `/root/galaxea_meta_backup_20260930.tar`。
- **Hy-Embodied**（`hy_embodied_convert.py`）：table_000 试跑通过；全量 22 张表转换 + 扫描在 dsw-4 跑（`/root/owam_hy/hy_full.sh`，日志 `/root/openwam_g1_logs/hy_full.log`，产物 `/root/Hy-Embodied-lerobotv3`）。完成后：审查扫描 → 删除 → 复制到 OSS → Notion。
- **lingbot-GM-100**：subagent 在 dsw-4 做（代码在它的 worktree，未合并）。R1 Pro 已转（16,772 条 / 152.4 h，`/root/lingbot-lerobotv3`）；AgiBot G1、AgileX 只有关节角，按用户决定做 FK（进行中）。
- **InternData-A1 仿真**（lift2 + split_aloha，用户决定只用 `sim_updated_lerobotv30`，`sim`/`sim_updated` 是同批旧版不用；真机 `physical/*` 是 v2.1 纯关节、共 255 条，暂不做）：225 个 tar 已解压到 OSS `/mnt/data/datasets/InternRobotics/InternData-A1-lerobotv3/<类别>/<本体>/<任务>/`（dsw-3 `/root/owam_intern/extract.sh`，按 `.<任务>.done` 续跑），格式即 `interndata_a1` 读取器所需，无需转换。
  - `episode_quality` 支持 A1：位姿取 `states.{left,right}_ee_to_robot_pose`（xyz + 四元数 wxyz → rot6d），夹爪用读取器的 `resolve_gripper_scale` 缩放到 [0,1]，头部相机 `images.rgb.head`，bucket 递归查找（bucket 名 = 相对路径）。
  - 官方 episodes 清单在数据文件边界把 episode 记到前一个文件（数据本身完整）；读取器早已按 `dataset_from_index` 对照实际行数重定位，扫描也照做（仅 A1）。
  - 试扫 2 个 bucket：lift2 0 命中；split_aloha close_the_microwave_left_arm 标出 16%（`pos_jump`/`rot_jump` 各约 340，`too_short` 58 条，最短 8 帧）。跳变已核对行对齐无误，是真实数据问题：工作臂 8–55 帧单步 > 5 cm，最大 0.64 m/帧。阈值不变。
  - 全量扫描在 dsw-4 跑：`/root/owam_intern/repo`，日志 `/root/openwam_g1_logs/intern_quality.log`，输出 `/root/intern_quality/`（未 apply）。多 episode 共用文件，`--delete` 不适用，按用户决定用 `--apply` 写黑名单，之后重算 `interndata_a1_stats_computation`。
- 基础设施：dsw-2 ossfs 挂载断开（需平台重挂），dsw-1、dsw-5 SSH 被拒（dsw-1 上有 G1 mid-train，需确认）。AgiBot 第二轮提取在 dsw-3 重跑（日志 `agibot_extract3.log`），完成后对 60 个不完整任务重转并补扫描、删除、同步 OSS。

## 双臂数据集指令质检（2026-10-09）

- 全量导出 `/mnt/data/filter_datasets` 17 个数据集的指令（3574 个 bucket、约 9.4 万条）：dsw-8 本地 `/root/task_dump.csv`（列 ds / bucket / task_index / task / n）。
- `episode_quality.prompt_ok` 新增 `EMPTY_SLOT`：冠词后直接跟介词或标点（如 "move the  to the book"）判为 `bad_prompt`。在全量导出上只命中 InternData-A1 仿真 5266 集，其他数据集 0 误报（"letter A." 这类已排除）。
- 不把 slug（lingbot 全部、Sim1 `fold_mat`）、资产 ID（`microwave_gr`、`Galbot_G1_…_new1`）加进 `bad_prompt`：`--apply` 会把命中集写进排除名单，等于整批丢数据；这些问题改在 loader 里做文本规范化（未做）。
- **数据修复已执行（2026-10-09，dsw-3）**，只改 `filter_datasets` 副本，源数据不动；改前 meta 备份 `/mnt/data/filter_datasets_v21_old/prompt_fix_20261009/{unitree_meta,interndata_meta}.tar`：
  1. `Unitree_G1_Dex3/G1_Dex3_GraspSquare_Dataset`（301 集）：原 "camera packaging"（README 也抄 BlockStacking），看视频是在黑胶带上从下到上叠红、黄、绿方块；`tasks.parquet` 和 `meta/episodes` 都改为 "Stack the three cubes on the black tape from bottom to top: red, yellow, green."。
  2. `Unitree_G1_Dex3/G1_Dex3_Pick{Apple,Bottle,Charger,Doll,Gum,Snack,Tissue}_Dataset`：`meta/episodes` 的 tasks（原全是 "Pick up the red cup on the table."）改成 `tasks.parquet` 文本（loader 实际读的，截帧核对正确）。
  3. `Unitree_G1_Dex1/G1_Dex1_MountCameraRedGripper_Dataset`：episodes "mount camera" → "mount camera."，与 tasks.parquet 一致。
  4. InternData-A1 仿真：`EMPTY_SLOT` 命中 5266 集（6 个 `continues_pick_and_place` 桶），32 集原已排除，新增 5234 集 / 69.98 h 写入 `excluded_episodes.json`（原因 `bad_prompt`）。仿真 415,085 → 409,851 集、2797.1149 → 2727.1349 h；Notion 数据集表已更新。**InternData 归一化统计待重算**（`interndata_a1_stats_computation`）。
  - 修复脚本只在 dsw-3 `/root/prompt_fix.py`（不进仓库，无参数为预览，`--apply` 写入）；回读核对通过，`load_episodes_parquet` 加载 3 个改过的 Unitree 桶正常。
- **指令规范化（2026-10-09 第二轮）**：新增 `openwam/dataloader/utils/prompt_text.py`（`normalize_prompt` + 共享正则）。`LeRobotV3Reader` 新增配置键 `normalize_prompt`（默认 false，因为 LIBERO / RoboTwin / G1D 等评测客户端发的是原始字符串）；`agibotworld.yaml`、`robocoin.yaml`、`interndata_a1.yaml` 设为 true。规则：去机器人名前缀（`Galbot_G1_`）和录制/版本后缀（`_new1`、`_0501_04`、`Copy`、` v3`、`·`），去两字母资产码（`microwave_gr` → microwave），整句是 slug 时连字符转空格，下划线一律转空格，去 please，合并空格，首字母大写，补句号。全量 dump 上 82,946 条指令中 36,893 条会被改写。用这些配置训练的模型，评测时也要把指令过一遍 `normalize_prompt`。
- **bad_prompt 扩展**：在空槽位之外加入重复词、slug、资产 ID（按用户要求）。全量 dump 上新增命中：重复词 ~1k 集（RoboTwin 509、RoboPro 332、IROS 116、RDT 40）；slug：lingbot 46,337（全部）、InternData 4,092、RoboCOIN 3,503、Sim1 1,976；资产 ID：InternData 59,556、RoboCOIN 3,791、Sim1 1,168、InternData 真机 200。**只是标记，没有对任何数据集跑 `--apply`**；slug/资产 ID 这两类 loader 已能规范化，`--apply` 前需决定是否真要排除。
- prompt 缓存：训练（`wan_backbone._train_prompt_embed_cache`）和部署（`prompt_embed_cache`）都是进程内 dict，没有落盘缓存，新进程自动按新文本重建，无需操作。
- **InternData 归一化统计已算（2026-10-09）**：按用户决定，把只有关节角、无 EEF 位姿的 `basic_tasks__lift2__store_the_toothbrushes_part{1,2}`（1,396 集，未排除 1,394 集 / 6.39 h）移到 `/mnt/data/filter_datasets_v21_old/InternData-A1_lerobotv3/`，记录在副本根目录 `deleted_buckets.json`；副本现 700 桶（lift2 377、split_aloha 323）。dsw-3 跑 `interndata_a1_stats_computation --split train` 生成 `filter_datasets/InternData-A1_lerobotv3/meta/stats_{lift2,split_aloha}.json`（3.08 亿 / 2.80 亿行，约 9 分钟，日志 dsw-3 `/root/intern_stats.log`）。用 `configs/dataloader/pretrain_data/interndata_a1.yaml` + 该 dataset_dir 构建完整 reader 通过：293,431,454 个训练窗口，prompt 已规范化，动作在 [-1, 1]。Notion：仿真 408,457 集 / 2720.74 h，整行 417,221 集 / 2800.35 h。
- lingbot-GM-100 副本用 AgiBotWorld reader 读会报缺 `action.ee_base`，目前没有能直接读它的 loader（与本次改动无关）。
- 其余发现（措辞、大小写、重复词 "with with"/"the the"、RoboPro "please"、RoboCOIN 942 条未用 task 与 32 集空 tasks、Galaxea tasks 表里的 `qualified`/`unqualified`）只记录，未处理。
- dsw-8 的 `/mnt/data` ossfs 在 2026-10-09 上午断开（传输端点尚未连接），需平台重挂。

## 单臂数据集清洗（2026-10-09 起）

- 范围（用户决定）：全部单臂、无移动/升降底盘的数据，含仿真（类型填「仿真」）；入库目录是 OSS `/mnt/data/filter_datasets`（用户说的 filtered_datasets 实际不存在）。Notion 库「单臂桌面操作数据集」（Data 质检 页下，结构同双臂库，类型里「单臂 UMI」），已写入 DROID success/failure、LIBERO 5 套件、Meta-World，状态「待确认」。
- 盘点结论（只读各数据集前几个桶的 `info.json`，不递归扫 OSS）：已是 v3 的单臂有 Cosmos3-DROID（success 57,639 集 / 346.13 h，failure 不纳入）、LIBERO（libero_90/10/goal/object/spatial，合计约 11.7 h）、Meta-World-MT50（仅 4 维 state、无视频，能否用待定）。DuoBench 是双臂；`OXE/droid_1.0.1` 与 Cosmos3-DROID 重叠，先不收。未判定、需转格式：OXE 单臂子集（`aloha_*` 双臂排除）、oxe-auge、RoboMIND、RoboMIND2.0 Franka/UR5、RoboCOIN 单臂桶、RoboFAC、ManiSkill-fail、10Kh-RealOmin、ManipArena；RoboCasa 带底盘大概率排除。G1D 自采里 `MoveibleLift` 本体带升降，不算。
- `episode_quality` 新增单臂：DROID（`observation.state.cartesian_position` xyz + 欧拉角、`gripper_position`）和 LIBERO（8 维 `observation.state` xyz + 轴角 + 两指位置，夹爪 = 指差 / 0.08），都转成 1 臂 xyz + rot6d，其余规则不变；头部相机加了 DROID `exterior_image_1_left` 和 LIBERO `image`。单测 `test_single_arm_droid_libero`；dsw-4 `pytest tests/dataloader/test_episode_quality.py` → 7 passed。
- 扫描在 dsw-4（代码 `/root/owam_single/repo`，源数据只读，未 apply）：LIBERO 带 `--video` → `/root/single_quality/libero`；DROID success 单进程、不带视频 → `/root/single_quality/droid`，日志 `/root/single_quality_droid.log`（并行是一个 bucket 一个进程，57k 集的单 bucket 会很慢；视频检查要不要补，看结果再定）。**结果还没看**。
- **已完成入库（2026-10-09）**：
  - LIBERO 5 套件 0 命中，原样复制到 `/mnt/data/filter_datasets/LIBERO_lerobotv3/<套件>`（4.2 GB，文件数/字节核对一致）。
  - DROID success 重扫带 `--video`（`--bucket-shards 24`，24 worker）：命中 3,508/57,639 集（`bad_prompt` 3,105 全是指令 `" |  | "`、`too_long` 1,204、`static` 111、`too_short` 54、`black_video` 26、`frozen_video` 20、`bad_video` 1）。用户决定：只排除空指令、`too_long` 也排除、补视频检查。`prompt_ok` 改为 " | " 连接的标注变体只要有一个可用即通过。剩 54,131 集 / 14,222,740 帧 / 263.38 h。复制到 `/mnt/data/filter_datasets/Cosmos3-DROID_lerobotv3/success`（628 GB，核对一致，含被排除集），在副本 `--quality droid_v2/quality.parquet --apply` 写 `excluded_episodes.json`（3,508，loader 读回 3,508）。**DROID / LIBERO 的归一化统计和读取器还没做**。
  - `episode_quality` 新增 `--bucket-shards N`（一个 bucket 的数据文件分给 N 个进程）。
  - Notion「单臂桌面操作数据集」：DROID success、LIBERO 5 套件已标「已完成」并填筛选后数据；OXE 35 个子集已排队（29 个单臂候选「待确认」、6 个移动/四足/双臂「未纳入」），都还没质检。
- **OXE 标准 Franka/UR5 五个已入库（2026-10-09）**：`OXE_SPECS` 按 bucket 名适配——stanford_hydra（xyz+欧拉，夹爪 state[7]）、taco_play（xyz+欧拉，夹爪 state[6]）、berkeley_autolab_ur5（xyz+四元数 xyzw 假设，夹爪取 action[6]）、utaustin_mutex / toto（7 个 Franka 关节角 → `franka_fk`，用 ready 位姿核对 TCP=(0.307,0,0.487)；mutex 夹爪 state[7]，toto 取 action[6]）。dsw-4 `--video` 扫描：运动指标正常（单步 0.7–2.7 cm，无跳变命中），只命中 `bad_video` 19 条；toto 唯一指令 "pour" 只有一个词，按规则全部 `bad_prompt`，决定保留。复制到 `/mnt/data/filter_datasets/OXE_lerobotv3/<bucket>`（共约 26 GB，核对一致），副本 `--apply` 写入排除。保留：ur5 997 集/5.43 h、hydra 567/9.89 h、taco 3,602/4.40 h、toto 993/2.98 h、mutex 1,498/5.02 h；Notion 已更新。utaustin_mutex 指令是带 `\n` 和 `tf.Tensor(b"...")` 包装的冗长改写，没清理。
- **2026-10-10 dsw-4 的 ossfs2 又被 OOM 杀掉**（我的 OXE 第二批 `--video --workers 12` 扫描时；dsw-4 上还有别人的 minimax_h3 训练在读 OSS）。需要平台重挂；dsw-4 本地 `/root/single_quality/{droid,droid_v2,libero,oxe5*}` 的结果不受影响。第二批 8 个 bucket（austin_sailor / jaco_play / kaist_nonprehensile / stanford_kuka_multimodal / berkeley_rpt / austin_sirius / austin_buds / cmu_play_fusion，`OXE_SPECS` 已加）**还没扫**：dsw-3 上重跑时发现 dsw-3 也有别人的 8 卡 FLUX 训练，已停掉自己的扫描，等选一台空闲机器、≤4 worker 再跑。原因与规矩已写进 `docs/h20.md`「ossfs2 会 OOM」一节。
- **OXE 第二批 8 个已入库（2026-10-10）**：`OXE_SPECS` 加了 austin_sailor / jaco_play / kaist_nonprehensile / stanford_kuka_multimodal（xyz+四元数 xyzw 假设；夹爪 state[7] 或 action[6]）和 berkeley_rpt / austin_sirius / austin_buds / cmu_play_fusion（Franka 关节角 FK）。扫描在 dsw-6（前 4 个）和 dsw-8（后 4 个）各 4 worker，没有 OOM（见上一条规矩）。
  - austin_sirius 540 集开头有 1–23 帧、austin_buds 若干集中间有最多 11 帧 **state 全 0 的填充帧**，FK 把它当真实姿态产生 0.8 m 假跳变；`_fill_zero_rows` 用最近有效帧代替后两个都 0 命中。**数据本身没改，训练读取要自己处理全 0 行**。
  - stanford_kuka：每集固定 50 帧、没有夹爪通道（action[6] 不是夹爪），`static` 会砍掉连续分布的低尾（405/3000），用户同意不按 static 排除（质检表里把 kuka 的 `effector_range` 置 1 后再 `--quality` 复用，只剩 pos_jump 14）。
  - 保留：sailor 239 集/4.88 h、jaco 1,073/2.15 h、kaist 196/0.77 h、kuka 2,986/2.07 h、rpt 906/3.63 h、play_fusion 576/13.11 h（类型按论文判为真机，未在数据里核实）、sirius 559/3.89 h、buds 50/1.90 h。都在 `/mnt/data/filter_datasets/OXE_lerobotv3/<bucket>`（复制核对一致，排除已 apply），Notion 已更新。OXE 副本现 13 个 bucket。
  - 还没处理的 OXE 单臂候选（state 语义没把握，保持「待确认」）：berkeley_fanuc / berkeley_mvp / dlr_edan / dlr_sara_grid_clamp / dlr_sara_pour / tokyo_u_lsmo / ucsd_kitchen / ucsd_pick_and_place / asu_table_top / nyu_franka_play / columbia_cairlab_pusht / stanford_robocook；无可用位姿的：cmu_franka_exploration（state 全 0）/ imperialcollege_sawyer / usc_cloth_sim（1 维 state）/ nyu_rot。
- **RoboMIND2.0 Franka / UR5 单臂已入库（2026-10-10）**：这两套是**双臂平台**（每集一个 ~380 MB HDF5，6 路相机 + 深度，`puppet|master/{arm,end_effector}_{left,right}_*_align`，JPEG 字节），用户决定只做纯粹单臂任务（双臂数据已在双臂筛选里做过）。盘点（`/root/robomind_survey.jsonl`，dsw-6，每任务第一集）：363 个任务、16.7 万集，单臂（只动一只臂）Franka 16 任务 / UR5 25 任务，另有 15 个 UR5 任务（5,780 集）HDF5 里没有 puppet 和图像，不可用。
  - 新增 `openwam/dataloader/utils/robomind2_convert.py`（+ `tests/dataloader/test_robomind2_convert.py`）：`select_tasks` 按第一集选单臂任务（名字含 both/two arm/dual 的丢），`convert_episode` 逐集再校验另一只臂静止（否则跳过并记入 `conversion_report.jsonl`）。输出每任务一个 bucket（`franka_<task>` / `ur5_<task>`）、agibotworld 布局；`observation.state.ee_base` 9 维（xyz + rot6d，四元数按 xyzw 读）、`action.ee_base` = 下一帧、`observation.state.gripper`（puppet）/`action.gripper`（master 指令）统一为 0 闭合 1 张开（源 0=张开 1=闭合，看腕部图像核对）、`observation.state.joints`；相机 `head`=camera_front（缩到 640×360）、`wrist`=动作臂腕部（640×480），不留深度。fps 只能估计（时间戳是整秒）：Franka 14、UR5 7，`info.json` 标 `fps_estimated`。
  - 转换：dsw-6（UR5 + 一半 Franka）、dsw-8（另一半 Franka）各 4 worker，没有 OOM。39 个任务成功，Franka 15 任务 26,811 集 / 389 万帧（77.28 h）、UR5 24 任务 5,715 集 / 57 万帧（22.63 h）；`franka_close_trash_can_lid` 没有腕部相机，转不了（代码已改成明确报 `missing_camera`，未重跑）；逐集跳过 not_single_arm：Franka 226、UR5 279（`ur5_place_donut_on_tray` 300 集里 273 集）。
  - 质检（本地盘 `--video --bucket-shards 4`，没有任何 black/frozen/bad_video 和 bad_prompt）：Franka 只 20 集（pos_jump 19、rot_jump 2）；UR5 在 20° 下命中 28%，多是单帧尖峰（20° 是按 30 Hz 定的，7 Hz 折算约 86°），用户决定 **UR5 旋转阈值 45°**、Franka 20°，位置阈值仍 0.10 m。应用后：Franka 26,791 集 / 77.23 h，UR5 4,919 集 / 19.02 h（剔除 796：rot_jump 640、pos_jump 156）。
  - 已复制到 `/mnt/data/filter_datasets/RoboMIND2.0-SingleArm_lerobotv3/<bucket>`（共约 32 GB，两台各自复制；核对 39 个 bucket 文件字节数一致——注意 `du -sb` 会因目录项大小差异误报不一致，要用 `find -type f -printf %s`），两份转换报告 `conversion_report_dsw{6,8}.jsonl`；在 OSS 副本上分别用 20°/45° `--apply`。dsw-6 / dsw-8 本地 `/root/RoboMIND2.0-single-lerobotv3` 和 `/root/rm_quality/` 仍在（实例释放会丢）。Notion 加了两行（Franka、UR5 单臂任务）。
  - 未做：RoboMIND2.0 的 `Franka-sim`、`Agilex`、`Ark`、`collection`；DROID / LIBERO / OXE / RoboMIND2.0 的归一化统计和读取器；`franka_close_trash_can_lid` 缺腕部相机，是否改成只用头部相机收下来待定。
- 踩坑：`--bucket-shards` 下某个分片为空时 `pd.concat` 会把 bool 列变 object，`~prompt_ok` 变成按位取反导致全部 `bad_prompt`；已改为只合并非空分片。已写出的 quality.parquet 如果是在修复前生成，需要先把 bool/int 列转回再 `--quality` 复用。
- 下一步：OXE 其余 24 个单臂候选（state/action 语义各异、5–10 Hz 低分辨率，逐个确认位姿列再加到 `OXE_SPECS`）；RoboMIND / RoboMIND2.0 Franka·UR5 / RoboCOIN 单臂桶 / oxe-auge 等要转格式；Meta-World 是否纳入待用户定；DROID 与 LIBERO 的 stats / reader。

## 夹爪事件子任务切分（2026-10-09）

- 目标（用户）：把 episode 按 action 轨迹切成子任务片段，每段结尾是一个关键点；**子任务 caption 先不做**。
- `openwam/dataloader/utils/gripper_segments.py`：只读 action 夹爪列（`action.{left,right}_gripper`，或 G1 扁平 16 维关节布局的 `action[14:16]`），跳过 `excluded_episodes.json`，输出每个 bucket 的 `<bucket>.json`（每集 `events` / `raw_events`，`(帧, L+|L-|R+|R-)`，`+` 开始合、`-` 开始松）和 `summary.json`。不改数据。
- 规则（逐条和用户在视频上核对过）：
  - 阈值：每集张开位 = 夹爪最大值，低于张开位 0.3 算闭合、回到 0.15 以内算张开；按 bucket 夹爪量程缩放（官方 4.5，部分仿真 / 纯关节桶 5.4，方向一致不用翻转）。
  - 边界 = 夹爪**开始变化**的帧（从阈值穿越点往回找，最多 1 s），合与松一致。
  - 空闲姿态：前 1 s 内开始且合到接近全闭（< 量程 10%）的闭合，连同它的张开都不算事件（官方 12.5% 的集开局主动合爪，PackBag 94%）。只合到一半的早期闭合是开局就夹着东西（自采 PourBeans 裁剪版），保留。
  - 短片段合并 `--min-seg 15`（0.5 s；1 s 会删掉分拣类任务里真实的快抓放）：同手松开后很快又合 = 重抓，两个事件都去掉；但间隔里夹爪张开到最大的是真放下，保留松开，只去掉后面那次短暂合爪；其余短片段并入前一段（第一段并入后一段）。
- 结论：所有 86 个官方桶都用夹爪；单手抓放、双手倒水类模板很稳（自采 5/6 任务主模式 52–100%）。不适合只靠夹爪的：擦拭类（Wipe_Board 等抓一次后主体动作在同一段）、叠衣服 / 装配（重抓多、顺序不固定，Fold_Clothes 1232 集 1192 种序列）。闭合到 0 不代表空抓（纸巾、纸杯沿、笔都会合到 0）。
- 结果（服务器本地，不在 OSS）：官方 dsw-4 `/root/seg/tool_out/`（parquet 副本 `/root/seg/g1dex1/`，27 GB），自采 dsw-8 `/root/seg/tool_out/`；叠加切分点的抽样视频 dsw-4 `/root/seg/vid*/`、dsw-8 `/root/seg/vid/`。官方 60,654 集：合并比例 5.8%，首事件为松开 2.8%（多是两手 0.5 s 内先后动作，并段后删掉的是前一个事件）。
- 仓库版相对探索脚本修了一个 bug：脚本算了“间隔 < 10 帧的闭合合并”却返回未合并结果。修后 4,028 集事件变化，抽查都是去掉假松开（ToolboxStorage ep1、ArrangePlates ep13）。
- 未做：子任务 caption、读取器按片段给提示词、PourBeans 按指令分模板、擦拭 / 叠衣服类的运动细分。

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

- 2026-10-09：新增 `gripper_segments`，按夹爪事件切子任务片段；跑完 86 个官方 G1-Dex1 桶和 G1D 自采数据。
- 2026-10-09：`episode_quality` 支持 DROID / LIBERO 单臂；单臂数据集 Notion 库建立，DROID 与 LIBERO 扫描进行中。
- 2026-09-30：InternData-A1 仿真 lift2/split_aloha 解压到 OSS；`episode_quality` 支持 A1 列格式与清单文件编号错位，全量扫描进行中。
- 2026-09-30：新增 Hy-Embodied、Galaxea 转换和 official/bag/body_motion 等清洗规则；Galaxea 桌面子集写入 OSS；Hy 全量转换、lingbot FK 进行中。

- 2026-09-29：G1D 自采数据转成 torso-EEF（URDF 正运动学，官方数据上误差 0），新增 `g1d_self` 读取器；准备后训练。
- 2026-09-29：AgiBot LeRobot v3 复制到 OSS（8.48 TB）并核对；原始数据路径迁到 `AgiBot/`，补下载随之迁移。
- 2026-09-29：`episode_quality` 新增 `--quality`（复用结果）和 `--delete`（物理删除，空 bucket 整体移除）；删除 2910 条不合格 episode，开始复制到 OSS。
- 2026-09-29：从 HF 补下载 AgiBot 缺失 tar 并补提取；转换支持灵巧手鱼眼腕部相机，报告按任务合并；重转 213 个任务并重新扫描。
- 2026-09-28：新增 AgiBotWorld-Beta 原始 → LeRobot v3 转换和规则式 episode 质量扫描；全量转换 12.8 万条，扫描标出 2.3%（未 apply）。

- 2026-09-27：分析训练速度（瓶颈在计算，不在数据），新增训练端 prompt 编码缓存。
- 2026-09-27：启动 G1 mid-train（全参，视频 / 动作分开设学习率，20000 步）。
- 2026-09-27：新增 G1-Dex1 多任务读取器、共享统计量、离线评估脚本；DSW 上 debug 微调和离线评估均已跑通。

## AtomBench-CobotMagic 转换与质检（2026-10-10）

- 用户决定：先做 AtomBench，GenieSim3.0 先不纳入（源数据仍是未解包的 tar.gz 分卷，filter_datasets 无副本，无质检产物）。
- `openwam/dataloader/utils/atombench_convert.py`：源 `/mnt/data/datasets/AtomBench-CobotMagic`（LeRobot v2.1，AgileX Cobot Magic 双 Piper 臂，15 任务 × 100 集，30 fps，每集每相机一个 H.264 mp4）→ 与 Galaxea 同布局的 v3 bucket（每任务一个 bucket、每集一个 parquet、mp4 原样复制不重编码、`meta/info.json` 最后写）。`python -m openwam.dataloader.utils.atombench_convert --src ... --out ... --workers 4`。
- 列：`observation.state.ee_base` 18 维 `[L_xyz, L_rot6d, R_xyz, R_rot6d]`；`action.ee_base` = 下一帧状态；`observation.state.gripper` / `action.gripper` `[L, R]` 取 `clip(raw,0,1)`，0=闭合 1=张开；源 `observation.state`（26 维：右臂 `[j1..6, 夹爪, xyz, rx,ry,rz]` 后接左臂）和 `action`（14 维 leader 关节，右臂在前）原样透传。相机：`image_top→head`、`image_left→hand_left`、`image_right→hand_right`（后两者是腕部相机，已看画面确认左右对应）。
- 核对：欧拉角是外旋 XYZ（`R=Rz·Ry·Rx`），用 Piper 正运动学（piper_sdk 的 DH）对关节角算位姿，位置误差 0.3 mm、旋转 Frobenius 误差 1e-3，所以直接用 `eef.euler_xyz_to_rot6d`。夹爪：全数据集 state 范围 ≈0..1、action 到 1.06（dm2 单任务只到 0.67–0.78，不能按单任务定标）；腕部相机在右手夹爪 0.78 时球刚落进篮子，0=闭合。
- **位姿坐标系是各臂自己的基座系**，左右两个基座之间的横向偏移数据里没有（两臂起始位置都约 (-0.01, 0, 0.28)），所以 18 维里左右两半不在同一个机器人坐标系下；`info.json` 的 `pose_frame` 已写明。要接进共用坐标系需要标定值。
- 源数据坑：dm3 / dm4 / di5 / di6 四个任务的 `frame_index` 从 43–86 起连续递增（源端裁了开头没重排），行数、timestamp、视频仍从 0 起且等于 `length`；转换时重排成 0..n-1，原起点记在 episodes 表 `source_frame_offset`。一开始检查写成"必须是 0..n-1"，这 4 个任务整体被跳过，已修。
- 结果：15 个 bucket、1500/1500 集、1,330,905 帧（12.32 h，与源一致），8.2 GB，无跳过。`episode_quality --video`（8 worker，本地盘）0 命中：最大单步位移 5.2 cm（阈值 10 cm）、最大单步旋转 15.2°（阈值 20°）、视频全部可解码，未 apply。产物 dsw-share1 本地 `/root/AtomBench-CobotMagic-lerobotv3`（含 `quality_summary.json`），质检输出 `/root/atom_quality/`，转换代码副本 `/root/owam_atom`。
- 已复制到 OSS：`/mnt/data/datasets/AtomBench-CobotMagic-lerobotv3` 和 `/mnt/data/filter_datasets/AtomBench-CobotMagic-lerobotv3`（`rsync -rL --size-only`）。`filter_datasets/MANIFEST.json` 是别人写的，没改。
- 验证：dsw-share1 `pytest tests/dataloader/test_atombench_convert.py tests/dataloader/test_galaxea_convert.py` → 4 passed（含 frame_index 偏移用例）。
- **未做**：没有 OpenWAM 读取器（Galaxea 也还没写）；没有算归一化统计；左右臂共用坐标系；GenieSim3.0。
- 踩坑（已存记忆）：并行 `du`/`find` 扫 OSS 会让 ossfs2 被 OOM 杀掉，整台机器挂载断开（2026-10-09 在 dsw-8 发生）；视频解码类扫描也有同样风险，worker ≤4、放在没人用的机器上。本次转换把输出写在本地盘，质检读本地盘，没有碰 OSS 的视频解码。

## GenieSim3.0 解包、转换与质检（2026-10-10）

- 用户决定（反转了先前"先不纳入"）：解包并处理，末端位姿按方案 A（从 state 推断，坐标系未验证）。
- 源：`/mnt/data/datasets/GenieSim3.0-Dataset/dataset_lerobot3.0/<任务>/g2_omnipicker/full/{meta,data,videos}.tar.gz.*`（65 任务，已是真正的 v3：多 episode 合并成大 parquet，视频按 `from/to_timestamp` 切片）。**2 个源文件在 OSS 上是 0 字节**：`tidy_up_workbench/videos.tar.gz.000`（上游 6.4 GB）和 `pull_drawer_number/data.tar.gz.000`（上游 35 MB）；ModelScope 下载这两个文件返回 HTTP 500，HF 镜像上没有 `dataset_lerobot3.0` 目录（`agibot-world/GenieSim3.0-Dataset` 存在但路径不同；GenieSimAssets 的目录接口 403），老格式 `dataset/` 里 `pull_drawer_number` 为空、`tidy_up_workbench` 是约 260 GB 的另一种格式。其余 196 个文件大小与上游逐一一致。这 2 个任务暂时排除，等上游修好再补。
- 解包（dsw-share1 本地盘 `/root/GenieSim3.0-unpacked/<任务>/`，脚本 `/root/genie_unpack2.sh`，4 路并行，`cat 分卷 | tar xzf -`，完成标记 `.done`）：63 个任务，90,703 条 / 17,742,247 帧 / 164.28 h，728 GB。任务名带括号的 8 个要用 `xargs -0` 传参，第一版脚本因此漏了。本地盘上的解包目录是**唯一副本**，还没复制到 OSS。
- `openwam/dataloader/utils/geniesim_convert.py`：原地给每个数据 parquet 加 4 列（原始列都保留，写临时文件再替换），`meta/info.json` 追加 features、`ee_pose_unverified`、`state_layout_inferred`，`geniesim_ee` 为完成标记。`python -m openwam.dataloader.utils.geniesim_convert --root /root/GenieSim3.0-unpacked`。
- **state 布局（186 维，无字段名，官方没有公开）是反推的，没有用 G2 运动学验证**：`state[14:17]`/`[17:20]` 左/右末端 xyz；`[126:135]`/`[135:144]` 左/右末端旋转矩阵（每一帧都正交、det +1，**按行优先**，转置没排除）；`[0]`/`[1]` 夹爪开度 0..120（官方 `omnipicker_reverse_relabel_gripper`，0=闭合）。左右顺序按末端顺序假设。action 40 维：`[0:2]` 夹爪指令（量纲不同，不用）、`[2:30]` 28 个关节目标（14 个臂关节在 `[16:30]`）、`[30:38]` 机身 8 维、`[38:40]` 底盘速度恒 0；**action 里没有末端位姿**。
- 位姿坐标系：起始帧末端位置在各任务间几乎一致（x≈0.62、y≈±0.40、z≈1.0），而机器人世界位置 `state[118:121]` 相差数米，所以 `[14:20]` 是**固定在机器人上的坐标系**（原点在底座下方地面，z 向上），不是世界系；部分任务因腰部姿态不同起始高度/伸出距离不同（z≈1.19、x≈1.1）。
- 新列：`observation.state.ee_base` 18 维 `[L_xyz, L_rot6d, R_xyz, R_rot6d]`（rot6d = R 的前两列）；`action.ee_base` = 下一帧状态；`observation.state.gripper` / `action.gripper` `[L, R]` = `clip(开度/120, 0, 1)`，0=闭合 1=张开；action 夹爪用下一帧状态（action[0:2] 量纲未核实）。
- 转换结果：63/63 任务成功，17,742,247 帧，**0 帧旋转矩阵不正交**；夹爪最大 0.999，左右末端位置对称、范围正常。
- `episode_quality`：`HEAD_CAMERAS` 加入 `observation.images.top_head`（GenieSim 头部相机名；之前会退回到第一个相机，即腕部）。`--video --workers 32 --bucket-shards 4` 不到 2 分钟扫完，输出 `/root/genie_quality/`（未 apply）：90,703 条标出 12,585（13.9%）——`bad_prompt` 12,125、`rot_jump` 397、`pos_jump` 208、`too_short` 1；视频检查全部通过。`bad_prompt` 是 4 个任务**整体**（`pick_block_number` 6476、`stock_in_the_supermarket` 3018、`pick_cup_size` 1943、`pick_accessory` 688）：`tasks.parquet` 的指令就是下划线任务名（slug），不是数据问题；`meta/info.json` 的 `instruction_segments` 里每条 episode 有真实指令，读取器可以用它。运动类（跳变、过短）共 606 条，最大单步位移 0.38 m、最大单步旋转 93°。
- **用户决定（2026-10-10）**：`bad_prompt` 那 12,125 条（4 个任务整体）不排除，指令就用下划线任务名（`tasks.parquet` 原样；读取器开 `normalize_prompt` 会把下划线转空格）。已 apply：把 `quality.parquet` 的 `prompt_ok` 全置 True 后 `--quality ... --apply`，排除 462 条（`rot_jump` 397 + `pos_jump` 208 + `too_short` 1，重叠后 462），写入 37 个任务的 `meta/excluded_episodes.json`（没有物理删除）。**筛选后 90,241 条 / 17,360,155 帧 / 160.74 h**。dsw-share1 `/root/genie_quality_keep/`。
- **复制到 OSS（用户要求）**：`/root/genie_copy.sh`（每任务一个 `rsync -rL --size-only`，6 路并行，先 `/mnt/data/datasets/GenieSim3.0-lerobotv3` 再 `/mnt/data/filter_datasets/GenieSim3.0-lerobotv3`，可断点续传，日志 `/root/genie_copy.log`，约 110 MB/s、每个目标约 2 小时）；`/root/genie_verify.py` 会在复制结束后逐文件核对两份（结果 `/root/genie_copy_verify.log` / `.json`）。**在核对通过前，dsw-share1 本地 `/root/GenieSim3.0-unpacked` 是唯一完整副本，不要删。**
- **未做**：读取器、归一化统计；用 G2 URDF 验证 EE 位姿（可用后重算）；`tidy_up_workbench` / `pull_drawer_number` 两个源文件等上游修复。
- 验证：dsw-share1 `pytest tests/dataloader/test_geniesim_convert.py tests/dataloader/test_episode_quality.py` → 12 passed。

