# QwenZone 常用命令(精简版)

> 只列你日常会用的。完整版见 git 历史。
> 工作目录:`/mnt/workspace/yama/starVLA`

---

## 🚀 训练(最常用)

```bash
cd /mnt/workspace/yama/starVLA
bash examples/Robotwin/train_files/run_qwenzone_train.sh
```

改训练步数 / GPU 数:
```bash
MAX_STEPS=50000 NUM_GPUS=2 bash examples/Robotwin/train_files/run_qwenzone_train.sh
```

**改训练哪个任务 / run 名**:编辑 `examples/Robotwin/train_files/starvla_qwenzone_robotwin.yaml` 里两处:
- `run_id: xxx`(决定输出目录名)
- `datasets.vla_data.data_mix: click_bell`(任务名,对应 `RoboTwin/data/<任务名>/`)

输出在 `results/Checkpoints_QwenZone/<run_id>/`,看 loss:
```bash
tail -f results/Checkpoints_QwenZone/<run_id>/train.log
# 或 SwanLab 网页:https://swanlab.cn/@yama/qwenzone
```

---

## 🎯 评估(最常用)

**训练完后,先做这一步**(每个新 run_dir 都要,复制反归一化配置):
```bash
cp results/Checkpoints_QwenZone/qwenzone_p1/dataset_statistics.json \
   results/Checkpoints_QwenZone/<run_id>/
```

然后跑评估:
```bash
cd /mnt/workspace/yama/starVLA
CKPT=results/Checkpoints_QwenZone/<run_id>/checkpoints/steps_50000_pytorch_model.pt \
TASK=click_bell TEST_NUM=10 MODE=demo_randomized \
  bash examples/Robotwin/eval_files/run_qwenzone_eval.sh
```

四个参数(都可选,有默认值):
| 参数 | 含义 | 例子 |
|------|------|------|
| `CKPT` | 要评估的 checkpoint | `results/.../steps_50000_pytorch_model.pt` |
| `TASK` | RoboTwin 任务名 | `click_bell` / `beat_block_hammer` |
| `TEST_NUM` | 评估几个 episode | `10` |
| `MODE` | `demo_randomized`(复杂,训练数据同款)或 `demo_clean`(干净) | 训练用 randomized 就 eval randomized |

输出在 ckpt 同目录的 `robotwin_eval_logs/`,视频命名 `episode{N}_success.mp4` / `episode{N}_fail.mp4`。

---

## 🔧 tmux 后台运行(避免断连)

```bash
tmux new -s train           # 创建会话(或 tmux a -t train 重连)
# 训练
bash examples/Robotwin/train_files/run_qwenzone_train.sh
# Ctrl+B D 退出(任务继续跑)

# 评估:另开 window
tmux new-window -t train -n eval
CKPT=... TASK=click_bell TEST_NUM=10 MODE=demo_randomized \
  bash examples/Robotwin/eval_files/run_qwenzone_eval.sh
```

---

## ❓ 常见报错

| 报错 | 解决 |
|------|------|
| `Missing dataset_statistics.json` | 上面的 cp 命令没做 |
| `unnorm_key='new_embodiment' not in` | `deploy_policy.yml` 改 `unnorm_key: robotwin_qwenzone`(已改,别动) |
| DeepSpeed OOM | 至少 2 张 80G 卡;别单卡跑 |
| 评估卡住不动 | server 加载 ~90s,等日志出现 `server listening` |
| 模型输出乱码 | 检查 `transformers==4.57.0`(`pip show transformers`) |

---

## 📍 关键文件(要改代码时看)

- 训练配置:`examples/Robotwin/train_files/starvla_qwenzone_robotwin.yaml`
- 模型主体:`starVLA/model/framework/VLM4A/QwenZone.py`
- 动作头:`starVLA/model/modules/action_model/ZoneMemory_ActionHeader.py`
- Stereo:`starVLA/model/modules/action_model/StereoEncoder.py`
- 数据:`starVLA/dataloader/gr00t_lerobot/hdf5_robotwin_dataset.py`

---

## 📊 当前各版本成绩(对比用)

| 版本 | 配置 | click_bell randomized 成功率 |
|------|------|------|
| t4(5万步) | T_obs=4, stereo 8 token, vis 池化 8, cmd 4 | 20% |
| t4v2(5万步) | T_obs=4, stereo 64 token(cat+ResConv), vis 不池化, cmd 16 | 30% |
