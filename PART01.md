
## 2. ARCHITECTURE VARIANTS

### 2.1 MLP-only ActorCritic (rl/actor_critic.py)

`python
class ActorCritic(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_sizes=[256, 256]):
        layers = [Linear(prev, h), ReLU() for h in hidden_sizes]
        self.trunk = Sequential(*layers)
        self.actor_head = Linear(hidden[-1], n_actions=45)
        self.critic_head = Linear(hidden[-1], 1)
`

**Параметры**: obs_size=209, actions=45, hidden=[256, 256]  
**Count params**: ~131K parameters total

### 2.2 CNN ActorCritic (rl/actor_critic_cnn.py)

`python
class ActorCriticCNN(nn.Module):
    cnn = Sequential(
        Conv2d(8, 32, 3), ReLU(),
        Conv2d(32, 64, 3), ReLU(),  
        Conv2d(64, 64, 3), ReLU(),
        AdaptiveAvgPool2d(1), Flatten()
    )
`

**Параметры**: n_channels=8, grid=29x29, actions=45  
**Count params**: ~152K parameters (CNN dominates)

### 2.3 Hybrid ActorCritic (rl/actor_critic_hybrid.py)

`python
class ActorCriticHybrid(nn.Module):
    flat_proj = Linear(obs_size=209, 256)
    cnn_proj = Linear(64*7*7, 256)       # CNN downsampled to 7x7
    trunk = Sequential(Linear(256, 512), ReLU(), 
                       Linear(512, 512), ReLU())
`

**Параметры**: obs_size=209, channels=8, grid=29  
**Count params**: ~1.49M parameters (largest)