# vit token遮蔽参数说明

## 参数设计

### 双参数设计
```yaml
action_model:
  vis_mask_probability: 0.5    # 遮蔽概率：每次训练step应用遮蔽的概率
  vis_mask_ratio: 0.3           # 遮蔽比例：应用遮蔽时，遮蔽token的比例
```

### 参数作用机制

**vis_mask_probability (遮蔽概率)**
- 范围：0.0 - 1.0
- 作用：控制每次前向传播时是否应用vit token遮蔽
- 值为0：不应用遮蔽（正常训练）
- 值为1：每次都应用遮蔽

**vis_mask_ratio (遮蔽比例)**
- 范围：0.0 - 1.0  
- 作用：当应用遮蔽时，随机遮蔽多大比例的vit token
- 值为0：不遮蔽任何token
- 值为1：遮蔽所有vit token（动作头完全不使用视觉）

## 设置示例

### 1. 温和正则化（推荐）
```yaml
vis_mask_probability: 0.5    # 50%的step应用遮蔽
vis_mask_ratio: 0.3           # 遮蔽30%的vit token
```
**效果**：每个训练step有50%概率随机遮蔽30%的视觉token

### 2. 激进正则化
```yaml
vis_mask_probability: 0.8    # 80%的step应用遮蔽
vis_mask_ratio: 0.5           # 遮蔽50%的vit token
```
**效果**：强制模型减少对视觉的依赖，但可能影响初期性能

### 3. 极端测试（完全禁用视觉）
```yaml
vis_mask_probability: 1.0    # 每次都应用遮蔽
vis_mask_ratio: 1.0           # 遮蔽100%的vit token
```
**效果**：动作头完全不使用视觉信息，只依赖cmd/state/stereo

### 4. 正常训练（无遮蔽）
```yaml
vis_mask_probability: 0.0    # 不应用遮蔽
vis_mask_ratio: 0.0           # 不遮蔽
```
**效果**：正常训练，作为baseline对比

## 推理时行为

推理时这两个参数都为0，不应用任何遮蔽：
```python
# 推理代码自动设置
vis_mask_probability = 0.0
vis_mask_ratio = 0.0
```

## 实验建议

### 阶段1：训练对比（2万步）
```yaml
# 实验组A：无遮蔽
vis_mask_probability: 0.0
vis_mask_ratio: 0.0

# 实验组B：温和遮蔽
vis_mask_probability: 0.5
vis_mask_ratio: 0.3

# 实验组C：激进遮蔽
vis_mask_probability: 0.8
vis_mask_ratio: 0.5
```

### 阶段2：根据结果调整
- 如果组B性能接近或优于组A：提高遮蔽强度
- 如果组B性能明显下降：降低遮蔽强度
- 目标：找到平衡点和最佳遮蔽策略

## 预期效果

### 短期（1-2万步）
- Loss可能略有波动
- 模型学习适应部分视觉缺失

### 中期（2-4万步）
- Loss稳定下降
- 模型学会利用其他模态信息

### 长期（4-5万步）
- 整体性能稳定
- 对视觉噪声/遮挡更robust
- 多模态利用更均衡

## d_max vs vit masking

d_max控制训练时的动作延迟，vit masking控制视觉依赖：
- `d_max=8`：训练时异步延迟8步
- `vis_mask_probability=0.5, vis_mask_ratio=0.3`：50%概率遮蔽30%视觉

两者配合：让模型学会在不完整信息下做决策
