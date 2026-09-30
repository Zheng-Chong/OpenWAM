# G1-Dex1 真机操作手册（倒红豆 / desk）

面向在机器人旁边的操作员。策略跑在远端 GPU 上，机器人这边只跑客户端。

**急停必须在手。** 下面的软件限幅只拦越界指令和速度突变，它拦不住一条"在训练分布内、但对眼前这个场景是错的"轨迹——那种情况机器人会很自信地做错事。

---

## 0. 开跑前的场景准备

这三条不满足，模型的表现没有参考意义：

- **底盘驻车，升降柱固定在采集时的高度。** 策略里完全没有这两个自由度（训练时就被排除了），它假定它们不动。
- **物体摆放照采集时来**：方口杯装红豆、灰色粗口杯做接收，位置和采集时大致一致。
- **三个相机不能动过**（头部左目 + 双腕）。分辨率、朝向、安装位置都必须和采集时一样。

每次 trial 记录一下物体摆放——真机没有仿真器帮我们随机化，这是事后能复现的唯一依据。

---

## 1. 通网检查（不上电也能做）

在机器人上：

```bash
curl http://$G1D_GATEWAY_HOST:10076/healthz
```

期望输出 `OK`。如果超时或连不上，**先别往下走**，是网络或服务端的问题，找算法侧。

## 2. 协议冒烟（不上电也能做）

```bash
cd /home/unitree/client/jepa_client
python motus_client_adapter.py --server_host $G1D_GATEWAY_HOST --server_port 10076
```

期望看到：

```
action_horizon: 12
cameras: ['head_left', 'wrist_left', 'wrist_right']
max_step_rad: 0.05
gripper limits: left (0.62, 5.40) right (3.82, 5.40)
actions (12, 16) float32  server infer ~190 ms
```

`server infer` 在 200ms 上下是正常的。**如果这一步整体花了好几秒**，说明链路慢，记下来告诉算法侧（多半要降图像质量或把服务端搬近）。

## 3. 机器人上电、进入初始位姿

按你们原来跑 MotusV2 的流程上电和使能——这部分没有改动，用的是同一套 `unitree_lerobot`。

启动：

```bash
cd /home/unitree/client/jepa_client
LEROBOT_ROOT=/home/unitree/unitree_lerobot \
UNITREE_DDSINTERFACE=eth0 IMAGE_HOST=192.168.123.164 \
bash run_g1d_jepa_client.sh
```

**强烈建议第一次带上初始位姿**，让双臂先走到采集集的第一帧再开始：

```bash
INIT_POSE_JSON=/path/to/episode/data.json bash run_g1d_jepa_client.sh
```

不带这个参数的话，机器人会从当前姿态直接开始。如果当前姿态离演示起点很远，开头一两秒会看到手臂在"追"目标位置——那是限幅器在起作用，不是故障，但起点差太多会让这次 trial 没意义。

## 4. 启动提示

屏幕会停在：

```
Enter 's' to start evaluation:
```

**这一刻手臂还没动。** 确认以下几点再回车：

- 急停在手
- 人不在手臂工作范围内
- 物体摆好了
- 上面的日志里 `instruction` 是 `把方口杯里的红豆，倒进灰色的粗口杯里，倒半杯`

按回车就开始（输入什么都一样，按回车即启动）。

## 5. 跑起来之后看什么

正常日志长这样：

```
[bootstrap] server=193ms chunk=(12, 16)
[splice] server=190ms stale=2 queue=10
[30] queue=7
```

- **`stale=` 是链路健康度。** 个位数正常。如果经常 `stale=` 超过 10，说明推理结果回来时已经过期大半，动作会变顿挫。
- **`whole chunk stale`** 这条警告意味着链路已经跟不上了，手臂基本只能执行每个 chunk 的最后一步。出现就停下来找算法侧。
- **手臂应该是连续运动的。** 如果是"走一下停一下"，同样是链路问题。

## 5.5 第一次连上后，立刻看一眼录下来的图

**这一步只需要做一次，但不做的话有两类错误永远发现不了。**

跑完第一次（哪怕只跑了几秒）后，在 GPU 侧找到最新的录制目录，打开这三张图：

```
recordings/episode_<时间戳>/000000_head_left.jpg
recordings/episode_<时间戳>/000000_wrist_left.jpg
recordings/episode_<时间戳>/000000_wrist_right.jpg
```

确认两件事：

- **颜色正常**——桌子是白的、木地板是棕的、机器人底座 LED 是蓝的。如果整体偏蓝、人的皮肤发青，那是红蓝通道反了（客户端用 PIL 而不是 cv2 编码 JPEG 导致）。
- **左右腕没有互换**——`wrist_left` 应该是左手看到的画面。可以手动只动左臂再抓一帧来确认。

这两类错误**都不会报错，也不会体现在动作数值上**，只会让机器人表现得"有点笨"。查一次，之后就不用再查了。

## 6. 停止

`Ctrl-C`。结束时会打印一行：

```
Done: 3 commands range-clamped, 17 rate-limited
```

这两个数值得记一下：

- **`range-clamped` 数量大**：模型在往训练数据没覆盖的关节角度上跑，说明场景和采集时差得比较多。
- **`rate-limited` 数量大**：模型想让手臂动得比限速快，通常是它对当前场景不确定。

**注意 `Ctrl-C` 之后手臂会保持在最后一条指令的姿态，不会自动松力。** 和你们原来的流程一样，按原来的方式收尾。

---

## 什么时候该立刻拍急停

- 手臂朝人、朝桌面或朝自身快速运动
- 出现明显的往复抖动（一般是链路太慢导致在追过期目标）
- 夹爪在没有物体的位置反复开合
- 屏幕刷出 Traceback 但手臂还在动

拍完急停后把终端最后 30 行截给算法侧。

---

## 常见问题

**`the server has action ensembling on ...`**
服务端配置不对，需要算法侧用 `--no-action-ensemble` 重启。客户端是故意不往下跑的——那种组合会让动作静默错位。

**`missing observation.images.cam_left_wrist ...`**
相机没起来或键名不对。检查 `IMAGE_HOST` 和相机服务。

**`WebSocket handshake failed` / 连接超时**
回到第 1 步。服务端可能停了或者隧道断了。

**`ModuleNotFoundError: No module named 'msgpack'`**
`pip install msgpack`。这是唯一新增的依赖。

**手臂完全不动，但日志在刷**
看 `queue=` 是不是一直 0，以及 `stale` 是不是巨大。多半是链路问题。
