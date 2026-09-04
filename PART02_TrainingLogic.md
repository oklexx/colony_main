
## 3. PPO HYPERPARAMETERS (rl/config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 3e-4 | Gradient step size |
| n_steps | 4096 | Rollout length per env |
| batch_size | 8192 | Update batch size |
| n_epochs | 10 | Epochs per update |
| gamma | 0.995 | Discount factor |
| gae_lambda | 0.98 | GAE smoothing |
| clip_range | 0.2 | PPO clipping width |
| ent_coef | 0.005 | Entropy coefficient ⚠️ |
| vf_coef | 0.5 | Value function weight |
| max_grad_norm | 0.5 | Gradient clipping |

## 4. OBSERVATION MODES

### Flat Mode:
- Shape: [batch, 209]
- Contains: money, taxes, population, resources, action masks
- Network: MLP only

### Minimap Mode:
- Shape: [batch, 8 channels, 29x29]
- Spatial grid with land/building types
- Network: CNN trunk + MLP

### Hybrid Mode:
- Dual input: flat (209D) + minimap (8x29x29)
- Fusion: element-wise add after projection
- Network: Both branches → merged trunk