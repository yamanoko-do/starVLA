# QwenZone 训练与评估命令手册

> memory-token VLA(block-wise attention + memory token + stereo encoder + transformer action head)
> 训练数据:RoboTwin 2.0(原始 HDF5 格式)

## 环境

| conda env | 路径 | 用途 |
|-----------|------|------|
| `starVLA` | `/mnt/workspace/yama/miniconda3/envs/starVLA` | 训练 + policy server(含 QwenZone/OpenStereo) |
| `RoboTwin` | `/mnt/workspace/yama/miniconda3/envs/RoboTwin` | eval client(RoboTwin 仿真) |

---

## 1. 训练

```bash
cd /mnt/workspace/yama/starVLA
bash examples/Robotwin/train_files/run_qwenzone_train.sh
```

自定义:
```bash
NUM_GPUS=2 MAX_STEPS=5000 bash examples/Robotwin/train_files/run_qwenzone_train.sh
```

- **框架**:starVLA trainer + DeepSpeed ZeRO-2
- **输出**:`results/Checkpoints_QwenZone/qwenzone_p2/`
- **监控**:SwanLab(project=qwenzone)+ `${OUTPUT_DIR}/train.log`
- **checkpoint**:`${OUTPUT_DIR}/checkpoints/steps_<N>_pytorch_model.pt`(每 save_interval)

> ⚠️ 训练前确认 GPU 空闲(`nvidia-smi`),DeepSpeed 单卡会 OOM(需 ≥2 卡或 CPU offload)。

---

## 2. 训练后、评估前:复制 dataset_statistics.json

eval 的 `from_pretrained` 需要 `run_dir/config.yaml` + `run_dir/dataset_statistics.json`。
config.yaml 训练自动保存,但 **HDF5 dataset 不自动保存 dataset_statistics**(需手动复制):

```bash
cp results/Checkpoints_QwenZone/qwenzone_p1/dataset_statistics.json \
   results/Checkpoints_QwenZone/qwenzone_p2/
```

> dataset_statistics.json 内容固定(action min/max=[-3,3] joints + [0,1] gripper;state 不归一化所以 dummy),所有 run_dir 通用。

---

## 3. 评估

```bash
cd /mnt/workspace/yama/starVLA
bash examples/Robotwin/eval_files/run_qwenzone_eval.sh
```

自定义(ckpt / 任务 / episode 数 / 模式):
```bash
CKPT=results/Checkpoints_QwenZone/qwenzone_p2/checkpoints/steps_5000_pytorch_model.pt \
TASK=beat_block_hammer TEST_NUM=10 MODE=demo_clean \
  bash examples/Robotwin/eval_files/run_qwenzone_eval.sh
```

- 启动 **policy server**(starVLA env,加载 QwenZone ckpt,含 stereo encoder)
- 启动 **RoboTwin eval client**(RoboTwin env,仿真)
- **输出**:`${ckpt_dir}/robotwin_eval_logs/<run>/`(server.log + eval.log + 视频)
- 视频命名:`episode{N}_success.mp4` / `episode{N}_fail.mp4`

---

## 4. 关键路径

| 用途 | 路径 |
|------|------|
| 训练 yaml | `examples/Robotwin/train_files/starvla_qwenzone_robotwin.yaml` |
| 训练脚本 | `examples/Robotwin/train_files/run_qwenzone_train.sh` |
| 评估脚本 | `examples/Robotwin/eval_files/run_qwenzone_eval.sh` |
| 框架 | `starVLA/model/framework/VLM4A/QwenZone.py` |
| 块状注意力 | `starVLA/model/framework/VLM4A/block_attention.py` |
| 动作头 | `starVLA/model/modules/action_model/ZoneMemory_ActionHeader.py` |
| Stereo 编码器 | `starVLA/model/modules/action_model/StereoEncoder.py` |
| 数据集(HDF5) | `starVLA/dataloader/gr00t_lerobot/hdf5_robotwin_dataset.py` |
| VLM base(含 🧠🌱) | `playground/Pretrained_models/Qwen3-VL-4B-Instruct-MemoryAction` |
| Stereo 权重 | `/mnt/workspace/yama/OpenStereo/output/MultiDataset/WAVEStereo/wavestereo_mixdataset/20260623_wavestereo_filter/ckpt/checkpoint_epoch_2.pth` |
| RoboTwin 数据 | `/mnt/workspace/yama/RoboTwin/data/<task>/demo_randomized/` |

---

## 5. 配置说明(yaml 关键参数)

```yaml
framework:
  qwenvl.base_vlm: .../Qwen3-VL-4B-Instruct-MemoryAction
  qwenzone.T_obs: 2          # 序列时间步
  qwenzone.N_act: 4          # 动作意图 token 数(🔍)
  qwenzone.N_mem: 4          # memory token 数(🧠)
  qwenzone.stereo:           # stereo encoder(冻结 WAVEStereo)
    ckpt_path / update_iters / pool_size / N_stereo_tokens
  action_model:
    state_dim: 28            # endpose(14) + joint(12) + gripper(2)
    action_dim: 14
    action_horizon: 50       # 动作 chunk 长度
    N_stereo: 8              # 动作头输入的 stereo token 数
trainer.eval_interval: 99999999   # 禁用 trainer MSE eval(chunk 模型用仿真 eval)
datasets.vla_data.dataset_py: robotwin_hdf5   # HDF5 dataloader(绕过 LeRobot)
```

---

## 6. 数据/动作约定

- **state(28 维,原始不归一化)**:`[left_endpose(7=pos3+quat4), right_endpose(7), left_arm(6关节角), right_arm(6), left_gripper(1), right_gripper(1)]`
  - 动作头的 `LayerNorm(state_dim)` 处理尺度差异
- **action(14 维,归一化)**:`[L6, R6, Lg, Rg]` 顺序(min_max joints→[-1,1],binary gripper→{0,1})
  - RoboTwin `joint_action/vector` 是 `[L6, Lg, R6, Rg]` 顺序,dataset 内部已 reorder 到 starVLA `[L6, R6, Lg, Rg]`
- **stereo**:`head_camera_left` + `head_camera_right` 双目(embodiment aloha-agilex)
- **VLM 视角**:仅 `head_camera_left`(单目)
- **unnorm_key**:`robotwin_qwenzone`(`deploy_policy.yml`)
- **embodiment**:`aloha-agilex`(`RoboTwin/task_config/demo_clean.yml`,双目)

---

## 7. 手动命令(不用脚本)

### 训练(手动 accelerate)
```bash
cd /mnt/workspace/yama/starVLA
PYTHONPATH=/mnt/workspace/yama/starVLA \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
HF_HOME=/mnt/workspace/yama/oss_yama/cache/hf_cache/hub \
PATH=/mnt/workspace/yama/miniconda3/envs/starVLA/bin:$PATH \
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 2 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robotwin/train_files/starvla_qwenzone_robotwin.yaml \
  --output_dir results/Checkpoints_QwenZone/qwenzone_p2 \
  --trainer.max_train_steps 5000
```

### 评估(手动 start_eval.sh)
```bash
cd /mnt/workspace/yama/starVLA
STARVLA_PYTHON=/mnt/workspace/yama/miniconda3/envs/starVLA/bin/python \
ROBOTWIN_PYTHON=/mnt/workspace/yama/miniconda3/envs/RoboTwin/bin/python \
bash examples/Robotwin/eval_files/start_eval.sh \
  -m demo_clean -n qwenzone_eval \
  -c /mnt/workspace/yama/starVLA/results/Checkpoints_QwenZone/qwenzone_p2/checkpoints/steps_5000_pytorch_model.pt \
  -t 10 beat_block_hammer
```

### tmux 后台运行
```bash
tmux new -s train
# 训练
bash examples/Robotwin/train_files/run_qwenzone_train.sh
# 评估(新 window)
tmux new-window -t train
bash examples/Robotwin/eval_files/run_qwenzone_eval.sh
```

---

## 8. 常见问题

| 问题 | 解决 |
|------|------|
| DeepSpeed OOM | 用 ≥2 GPU;或改 `ds_config.yaml` 加 `offload_optimizer`(需 ninja + `zero_force_ds_cpu_optimizer:false`) |
| `unnorm_key='new_embodiment' not in [...]` | `deploy_policy.yml` 的 `unnorm_key` 改成 `robotwin_qwenzone` |
| `Missing dataset_statistics.json` | 从 qwenzone_p1 复制到新 run_dir(见第 2 节) |
| `KeyError: 'data'`(eval) | server predict_action 报错,看 `*_server.log` 的 traceback |
| timm/transformers 版本 | 必须 `transformers==4.57.0`(5.x 输出乱码) |
| eval 卡在 server 启动 | server 加载 4.7B+stereo 需 ~90s,等 `server listening` |
| trainer eval 报 predict_action 错 | chunk 模型不兼容 trainer MSE eval,已用 `eval_interval:99999999` 禁用 |

---

## 9. 验证过的里程碑

- ✅ 训练跑通(loss 0.545→0.0078,2 GPU DeepSpeed)
- ✅ 部署 eval 链路跑通(完整 episode 400 步无错误)
- ⚠️ 效果 fail(5 条数据过拟合)——需全量数据训练才能评估真实泛化
