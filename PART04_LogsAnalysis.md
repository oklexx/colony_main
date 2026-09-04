
## 6. LOG ANALYSIS (From run_012.log)

### Training Statistics:

`
[Step 10,485,760/10,485,760] 
FPS=24,912 | episodes=30,902 best_reward=4.41 
p_loss=-0.0021 v_loss=0.2240 ent=2.6059 KL=0.00740
GPU_mem=2440MB
`

### Interpretation:

| Metric | Value | Status |
|--------|-------|--------|
| FPS | ~24,900 steps/sec | ✅ Excellent |
| Best Reward | 4.41 | ⚠️ Low |
| policy_loss | -0.0021 | ✅ Converged |
| value_loss | 0.2240 | ✅ Stable |
| entropy | 2.6059 | ✅ Good diversity |
| KL divergence | 0.00740 | ✅ Normal (< 0.01) |
| GPU Memory | 2440MB / 17GB | ✅ 14% utilization |

### Episode Performance (From reward_debug.log):

`
STEP 1: total=15.4, novelty=15.0, surv=0.1 → ep_return grows negative
STEP 2-31: PRESERVE action repeated, reward=-0.6/step
STEP 50+: ep_return = -29.4 (FAILED)
`

### Problem Pattern:

Model repeatedly selects **PRESERVE** action for 50+ consecutive steps:
- Probability: ~0.48 (highest among actions)
- Reward per step: -0.6
- Episode returns decline linearly

**Root cause analysis**:
1. High preserve_penalty (-0.5) suppresses building exploration
2. Low uild_bonus (1.0) doesn't compensate idle penalties
3. Model fails to find optimal long-term building strategy