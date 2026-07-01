# QwenZone 常用命令(精简版)

> 只列你日常会用的。完整版见 git 历史。
> 工作目录:`/mnt/workspace/yama/starVLA`

---

## 🧠 Phase 2 是什么(一句话)

完整实现 `tt.md` 的设计:**双 Pass 训练**(Pass1 并行 + Pass2 RNN 串行)+ **异步延迟 δ**(动作意图可延迟)+ **RNN 推理**(模式二)。解决 Phase 1 两个缺陷:① 训练只见过 T=4 序列、推理 history 累积导致的分布偏移;② VLM 慢/动作头快的延迟未建模。

新增 4 个旋钮(在 yaml 的 `qwenzone` 段):
| 旋钮 | 含义 | 默认 |
|------|------|------|
| `lambda_rnn` | Pass2 (RNN) loss 权重;0 = 退化为 Phase 1 单 Pass | `0.1`(稳了可提 0.3) |
| `d_max` | 异步延迟上限,δ ~ U{t-d_max, …, t};0 = 无延迟 | `5` |
| `infer_mode` | 推理模式:`full`(模式一,完整历史)/ `rnn`(模式二,单步+记忆注入)。**只有这两种**,异步不再是独立模式 | `full` |
| `vlm_stride` | 异步 LLM 刷新步幅:`0`=每步都跑 LLM(同步);`>0`=每 K 步刷一次 LLM、中间步复用缓存意图(K-1 ≤ d_max)。**full 和 rnn 都支持**(命令行 `--vlm_stride` 传) | `0` |

> **为什么没有 `async` 模式了**:异步(VLM 慢/动作头快,意图可延迟)本质是训练时 `δ` 支撑的**运行策略**,不是独立模式。`vlm_stride=0` 即同步,`>0` 即异步,与 `infer_mode` 正交。

---

## 🚀 训练(最常用)

**Phase 2(默认推荐):**
```bash
cd /mnt/workspace/yama/starVLA
bash examples/Robotwin/train_files/run_qwenzone_train_phase2.sh
# 改步数/GPU:MAX_STEPS=50000 NUM_GPUS=2 bash ...run_qwenzone_train_phase2.sh
```
配置文件:`examples/Robotwin/train_files/starvla_qwenzone_robotwin_phase2.yaml`,输出 `results/Checkpoints_QwenZone/qwenzone_click_bell_v2phase2/`。

**Phase 1(旧,单 Pass,作 baseline):** `bash examples/Robotwin/train_files/run_qwenzone_train.sh`

**改任务 / run 名**:编辑 yaml 里 `run_id`(输出目录名)和 `datasets.vla_data.data_mix`(任务名,对应 `RoboTwin/data/<任务名>/`)。

看 loss:
```bash
tail -f results/Checkpoints_QwenZone/<run_id>/train.log
# SwanLab: https://swanlab.cn/@yama/qwenzone
# Phase 2 会多出 train/parallel_loss 和 train/rnn_loss 两条曲线(都应下降)
```

> 单元自测(不训练,验证双 Pass + 注入机制):`CUDA_VISIBLE_DEVICES=0 python -m starVLA.model.framework.VLM4A.QwenZone`。包含 hook 等价性测试(注入原 embedding 必须 0 偏差)、tiny 训练、RNN 推理冒烟。

---

## 🎯 评估(最常用)

**训练完后,先做这一步**(每个新 run_dir 都要,复制反归一化配置):
```bash
cp results/Checkpoints_QwenZone/qwenzone_p1/dataset_statistics.json \
   results/Checkpoints_QwenZone/<run_id>/
```

**单 GPU 评估(简单场景)—— 命名参数:**
```bash
cd /mnt/workspace/yama/starVLA

# 必填 --ckpt,其余有默认值。GPU 0, full 模式(同步,每步跑 LLM)
CUDA_VISIBLE_DEVICES=0 \
bash examples/Robotwin/eval_files/run_qwenzone_eval.sh \
  --gpu_id 0 --infer_mode full --vlm_stride 0 \
  --ckpt results/Checkpoints_QwenZone/qwenzone_click_bell_v2phase2_t8/checkpoints/steps_50000_pytorch_model.pt \
  --task click_bell --test_num 10 --mode demo_randomized

# rnn 模式 + 异步(每 5 步刷一次 LLM,中间步复用缓存意图):--infer_mode rnn --vlm_stride 5
```

参数说明:
| 参数 | 含义 | 默认 |
|------|------|------|
| `--gpu_id` | GPU ID(需配合 `CUDA_VISIBLE_DEVICES`,两者保持一致) | `0` |
| `--infer_mode` | 推理模式 | `full` / `rnn` |
| `--vlm_stride` | LLM 刷新步幅:`0`=同步(每步跑 LLM);`>0`=异步(每 K 步刷一次,中间步复用缓存意图,K-1≤`d_max`) | `0` |
| `--ckpt` | checkpoint 路径(必填) | — |
| `--task` | RoboTwin 任务名(也可作末尾位置参数) | `beat_block_hammer` |
| `--test_num` | 每个 task 的 episode 数 | `1` |
| `--mode` | 评估模式:`demo_randomized`(训练同款)/ `demo_clean` | `demo_clean` |

**⚠️ 双模式并行评估(Phase 2 核心:对比 full vs rnn) — 必须注意端口冲突！**

问题：两个评估默认都用端口 5694，会导致第二个评估启动失败（`OSError: [Errno 98] address already in use`）。

**解决方案（命名参数）：**
```bash
cd /mnt/workspace/yama/starVLA
CKPT=results/Checkpoints_QwenZone/qwenzone_click_bell_v2phase2_t8/checkpoints/steps_50000_pytorch_model.pt

# GPU 0: full 模式 (端口 5694)
CUDA_VISIBLE_DEVICES=0 ROBOTWIN_BASE_PORT=5694 \
bash examples/Robotwin/eval_files/run_qwenzone_eval.sh \
  --gpu_id 0 --infer_mode full --vlm_stride 0 \
  --ckpt $CKPT --task click_bell --test_num 10 --mode demo_randomized

# GPU 1: rnn 模式 (端口 5695，必须不同！)
CUDA_VISIBLE_DEVICES=1 ROBOTWIN_BASE_PORT=5695 \
bash examples/Robotwin/eval_files/run_qwenzone_eval.sh \
  --gpu_id 1 --infer_mode rnn --vlm_stride 0 \
  --ckpt $CKPT --task click_bell --test_num 10 --mode demo_randomized
```

> 一行起双评估(自动 tmux + 端口分配):`bash examples/Robotwin/eval_files/run_qwenzone_dual_eval.sh --task click_bell --test_num 10 --mode demo_randomized`(GPU0 rnn / GPU1 full;可加 `--vlm_stride 5` 让两边都开异步对比)。

**关键环境变量说明:**
| 环境变量 | 作用 | 示例 | 必要性 |
|----------|------|------|--------|
| `CUDA_VISIBLE_DEVICES=N` | 限制进程只能看到 GPU N | `CUDA_VISIBLE_DEVICES=0` | **必需** — 否则 `start_eval.sh` 会自动检测所有 GPU，两个评估会在同一 GPU 上跑 |
| `ROBOTWIN_BASE_PORT=N` | 设置推理服务器端口 | `ROBOTWIN_BASE_PORT=5695` | **并行评估时必需** — 默认 5694，两个评估不能用同一端口 |

**为什么 `--gpu_id` 不够，还要 `CUDA_VISIBLE_DEVICES`？**
- `--gpu_id` 只传给内部逻辑，但 `start_eval.sh` 会**自动检测所有可用 GPU**（`nvidia-smi --list-gpus`）。
- 必须用 `CUDA_VISIBLE_DEVICES` 真正限制 GPU 可见性，两者保持一致(`CUDA_VISIBLE_DEVICES=N` ↔ `--gpu_id N`)。

输出在 ckpt 同目录 `robotwin_eval_logs/`,`episode{N}_success.mp4` / `episode{N}_fail.mp4`。

> **每 episode 自动 reset**:eval 客户端每个 episode 开头会发 `reset` 给 server,清空 full 模式的 history 缓冲和 rnn 模式的 `m_state`(已接好:`reset_model → ModelClient.reset → server → framework.reset_history`)。所以同一任务的多个 episode 之间不会串记忆。

---

## 🔧 tmux 后台运行(避免断连)

**训练:**
```bash
tmux new -s train
bash examples/Robotwin/train_files/run_qwenzone_train_phase2.sh
# Ctrl+B D 退出(任务继续跑)
```

**双模式并行评估 (tmux 后台运行):**
```bash
# GPU 0: full 模式 (端口 5694)
tmux new-session -d -s full_eval \
  "cd /mnt/workspace/yama/starVLA && \
   CUDA_VISIBLE_DEVICES=0 ROBOTWIN_BASE_PORT=5694 \
   bash examples/Robotwin/eval_files/run_qwenzone_eval.sh \
     --gpu_id 0 --infer_mode full --vlm_stride 0 \
     --ckpt results/Checkpoints_QwenZone/qwenzone_click_bell_v2phase2_t8/checkpoints/steps_50000_pytorch_model.pt \
     --task click_bell --test_num 10 --mode demo_randomized"

# GPU 1: rnn 模式 (端口 5695)
tmux new-session -d -s rnn_eval \
  "cd /mnt/workspace/yama/starVLA && \
   CUDA_VISIBLE_DEVICES=1 ROBOTWIN_BASE_PORT=5695 \
   bash examples/Robotwin/eval_files/run_qwenzone_eval.sh \
     --gpu_id 1 --infer_mode rnn --vlm_stride 0 \
     --ckpt results/Checkpoints_QwenZone/qwenzone_click_bell_v2phase2_t8/checkpoints/steps_50000_pytorch_model.pt \
     --task click_bell --test_num 10 --mode demo_randomized"

# 查看所有会话
tmux ls

# 进入某个会话查看进度
tmux attach -t full_eval   # 或 rnn_eval
# 退出会话: Ctrl+B D (任务继续跑)

# 停止某个评估
tmux kill-session -t full_eval  # 或 rnn_eval
```

> **双评估核心要点**:
> 1. **`CUDA_VISIBLE_DEVICES=N`** — 限制进程只能看到 GPU N（必需！否则会检测到所有 GPU）
> 2. **`ROBOTWIN_BASE_PORT=N`** — 设置不同端口（必需！默认 5694，两个评估会冲突）

**输出目录自动带模式后缀**:
```
eval_result/click_bell/model2robotwin_interface/demo_randomized/qwenzone_eval/
├── 2026-06-28 17:23:34_full/   # full 模式的结果（自动从参数获取）
└── 2026-06-28 17:23:34_rnn/    # rnn 模式的结果（自动从参数获取）
```

> 脚本会自动从 `infer_mode` 参数（full/rnn）添加后缀，无需手动设置 `EVAL_TIME_SUFFIX`。

---

## ❓ 常见报错

| 报错 | 解决 |
|------|------|
| `Missing dataset_statistics.json` | 上面的 cp 命令没做 |
| `unnorm_key='new_embodiment' not in` | `deploy_policy.yml` 改 `unnorm_key: robotwin_qwenzone`(已改,别动) |
| DeepSpeed OOM | 至少 2 张 80G 卡;Pass2 会多一份 B*T 短序列 forward,显存吃紧就 `per_device_batch_size=1` |
| 评估卡住不动 | server 加载 ~90s,等日志出现 `server listening` |
| 模型输出乱码 | 检查 `transformers==4.57.0`(`pip show transformers`);5.x 会乱码 |
| RNN 模式 eval 串记忆 | 老版 eval client 不发 reset;用更新后的 `model2robotwin_interface.py` |

---

## 📍 关键文件(要改代码时看)

- **Phase 2 训练配置**:`examples/Robotwin/train_files/starvla_qwenzone_robotwin_phase2.yaml`
- **模型主体(双 Pass + 注入 + 两种推理模式 full/rnn + vlm_stride 异步)**:`starVLA/model/framework/VLM4A/QwenZone.py`
- 注意力掩码:`starVLA/model/framework/VLM4A/block_attention.py`(Phase 2 复用,无需新掩码 —— RNN 序列就是 T=1 的标准 block 序列)
- 动作头:`starVLA/model/modules/action_model/ZoneMemory_ActionHeader.py`
- Stereo:`starVLA/model/modules/action_model/StereoEncoder.py`
- 数据:`starVLA/dataloader/gr00t_lerobot/hdf5_robotwin_dataset.py`
- eval reset 链路:`deployment/model_server/tools/websocket_policy_{server,client}.py` + `policy_wrapper.py` + `examples/Robotwin/eval_files/model2robotwin_interface.py`

> Phase 2 的关键机制:**注入** = 在 `model.model.language_model` 上挂 forward_pre_hook,把上一步 M_t 的 hidden state 写进当前步 🌱(m_prev)位置的 `inputs_embeds`(此时图像特征已合并、mrope 位置已算好)。详见 QwenZone.py 的 `_injection_ctx` / `_rnn_inject_hook`。

---

## 📊 当前各版本成绩(对比用)

| 版本 | 配置 | click_bell randomized 成功率 |
|------|------|------|
| t4(5万步) | T_obs=4, stereo 8 token, vis 池化 8, cmd 4 | 20% |
| t4v2(5万步) | T_obs=4, stereo 64 token(cat+ResConv), vis 不池化, cmd 16 | 30% |
| **v2phase2** | t4v2 + Pass2(RNN)+ 异步δ(d_max=8, λ=1.0) + 双模式推理(full/rnn, `vlm_stride` 控制异步) | 待测 |

Phase 2 成功标准:full 模式 ≥ Phase1(≥30%);rnn 模式成功率接近 full + **抖动更小** + 单步延迟稳定(短序列、无 history 增长)。异步(`vlm_stride>0`)应在成功率不掉太多的前提下进一步降低单步延迟。
