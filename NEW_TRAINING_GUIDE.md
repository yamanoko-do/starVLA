# 带vit token遮蔽的训练启动指南

## 新功能说明
添加了随机vit token遮蔽功能，减轻动作头对rgb视觉的过度依赖。

## 修改内容
1. 动作头训练时随机遮蔽30%的视觉token
2. 推理时不遮蔽，保持完整性能
3. 通过正则化强制动作头利用其他模态信息

## 启动训练

### 基础训练（2张GPU，5万步）
```bash
cd /mnt/workspace/yama/starVLA
bash examples/Robotwin/train_files/run_qwenzone_train_phase2.sh
```

### 自定义训练
```bash
cd /mnt/workspace/yama/starVLA

# 修改遮蔽比例（默认0.3，建议范围0.1-0.5）
# 在 starvla_qwenzone_robotwin_phase2.yaml 中设置 vis_mask_ratio

# GPU 0: full 模式
CUDA_VISIBLE_DEVICES=0 bash examples/Robotwin/train_files/run_qwenzone_train_phase2.sh

# 2张GPU训练
NUM_GPUS=2 bash examples/Robotwin/train_files/run_qwenzone_train_phase2.sh
```

## 监控重点
1. **Loss曲线**: 观察train/parallel_loss和train/rnn_loss
2. **验证成功率**: 定期评估full和rnn模式
3. **过拟合风险**: 如果遮蔽比例过高(>0.5)，可能影响性能

## 预期效果
- 动作头更均衡地利用所有模态
- 减少对视觉信息的过度依赖
- 提升对本体感觉和动作意图的关注
- 可能略微降低初期性能，但长期更robust
