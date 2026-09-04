# 🏗️ 02-ARCHITECTURE.md — Общая архитектура UI улучшений

## 📊 Архитектурная диаграмма

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌────────────────────────────┐  │
│  │   EnvManager │────▶│   AsyncTrainer│────▶│      Worker Process        │  │
│  │              │     │               │     │                           │  │
│  │ - n_envs=8   │     │ - policy     │     │ - Training loop           │  │
│  │ - curriculum │     │ - rollouts   │     │ - LoopDetector            │  │
│  │ - stages[]   │◀────│ - update     │     │ - Curriculum tracking     │  │
│  └──────────────┘     └──────────────┘     │                           │  │
│                                            │                           │  │
│                             ┌──────────────┴───────┐                    │  │
│                             │      metrics_queue   │◀────────────────────┤  │
│                             │ (worker → UI)        │                     │  │
│                             └──────────────────────┘                     │  │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                              COMMUNICATION LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         train_ui/worker.py                            │  │
│  │                                                                       │  │
│  │  def run_train(cfg, queue, command_queue):                           │  │
│  │      p = Process(target=_worker, args=(queue, command_queue))        │  │
│  │      p.start()                                                        │  │
│  │                                                                       │  │
│  │  # Separate threads for processing:                                  │  │
│  │  - metrics thread: reads from queue → send to UI                     │  │
│  │  - commands thread: reads from command_queue → execute               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Main Window (UI)                               │  │
│  │                                                                       │  │
│  │  - Parameters Panel (PARAM_SPECS + REWARD_SPECS)                     │  │
│  │  - Curriculum Table Widget                                            │  │
│  │  - Status Bar:                                                         │  │
│  │    ├─ KLStatusWidget          ───┐                                    │  │
│  │    ├─ CurriculumProgressWidget ──┤                                    │  │
│  │    ├─ ActionLoopWidget         ──┤                                    │  │
│  │    ├─ ReturnStatisticsWidget   ──┤                                    │  │
│  │    └─ QuickActionsPanel        ──┘                                    │  │
│  │                                                                       │  │
│  │  - Message Processing Thread:                                         │  │
│  │    Queue.get_nowait() → decode → update widgets                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                              DATA FLOW                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      PROGRESS MSG (worker → UI)                      │  │
│  │  {                                                                      │  │
│  │    "type": "progress",                                                  │  │
│  │    "done": 100_000,                                                     │  │
│  │    "total": 1_000_000,                                                  │  │
│  │    "fps": 25.5,                                                         │  │
│  │    "best_reward": 150.3,                                                │  │
│  │    "policy_loss": 0.42,                                                 │  │
│  │    "value_loss": 0.18,                                                  │  │
│  │    "entropy": 2.87,                                                     │  │
│  │    "kl": 0.032,                                                         │  │
│  │    "top_actions": {"PRESERVE": 0.48, "BUILD_HOUSE": 0.12, ...},        │  │
│  │    "loop_detected": true,                                               │  │
│  │    "loop_action_name": "PRESERVE",                                      │  │
│  │    "curriculum_stage_active": 0                                         │  │
│  │  }                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      COMMAND MSG (UI → worker)                        │  │
│  │  {                                                                      │  │
│  │    "type": "command",                                                  │  │
│  │    "cmd": "boost_entropy",                                             │  │
│  │    "payload": {"factor": 2.0}                                           │  │
│  │  }                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Потоки данных

### Thread 1: Metrics Processing (UI thread)
```python
while True:
    try:
        line = metrics_queue.get_nowait().decode('utf-8')
        msg = P.decode(line)
        
        if isinstance(msg, P.ProgressMsg):
            update_kl_status(msg.kl)
            update_top_actions(msg.top_actions)
            show_loop_alert(msg.loop_detected)
    except Empty:
        pass
```

### Thread 2: Command Processing (UI thread)
```python
while True:
    try:
        line = command_queue.get_nowait().decode('utf-8')
        cmd_msg = P.decode(line)
        
        if isinstance(cmd_msg, P.CommandMsg):
            handle_command(cmd_msg.cmd, cmd_msg.payload)
            
            # Examples:
            # - boost_entropy: multiply ent_coef by factor
            # - reset_curriculum: set current_stage = 0
            # - pause/resume: toggle training state
    except Empty:
        pass
```

## 🧩 Модульная структура

### rl/loop_detector.py
```python
class LoopDetector:
    """Stateless detector that can be reused across multiple runs."""
    
    update(env_idx, action_idx, action_names) → Optional[str]  # action name if loop detected
    update_batch(env_indices, actions_tensor, action_names) → Dict[int, str]
    get_stats() → Dict[statistics]
    detect_and_boost_entropy(em, ppo, current_step) → bool
    
    Thresholds (configurable):
        - PRESERVE: 3 actions
        - BUILD_*: 5 actions  
        - DEMOLISH: 4 actions
        - DAY/WEEK: 15-20 actions (lenient)
```

### EnvManager extensions
```python
class EnvManager:
    add_curriculum_milestone(steps, stage_id)
    _should_advance_curriculum(current_step) → bool
    set_curriculum_stage(stage_id)
    get_allowed_buildings_for_stage(stage_id) → List[str]
    get_curriculum_progress(current_step=None) → Dict[progress_info]
```

## 🎨 UI Layout (MainWindow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Title Bar: Сахалинская колония — обучение моделей                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐  ┌─────────────────────────────────────────────────┐ │
│  │ Param Panel      │  │ Metrics Display                                 │ │
│  │ - total_steps   ▶ │  │ FPS:    ████████████░░░░░░  25.5               │ │
│  │ - n_envs        ▶ │  │ Best R: ████░░░░░░░░░░░░░░  150.3             │ │
│  │ - learning_rate ▶ │  │ Policy L: ███░░░░░░░░░░░░░   0.42             │ │
│  │ - ...           ▶ │  │ Value L: ████░░░░░░░░░░░░░    0.18            │ │
│  └──────────────────┘  ├─ Entropy: 2.87                                  │ │
│                         └─ KL:      0.032                                 │ │
│                            ████████░░░░░░ (target: 0.03)                 │ │
│                                                                             │
│  ┌──────────────────┐  ┌─────────────────────────────────────────────────┐ │
│  │ Curriculum       │  │ KL Status Widget                                 │ │
│  │ Stage: 0/3       │  │ ┌─────────────────────────────────────────────┐ │ │
│  │ [Prev] [Next]    │  │ │ STATUS: OPTIMAL ⭐                          │ │ │
│  │ Schedule:        │  │ ├─ KL Progress Bar (0.01-0.08 scale)          │ │ │
│  │  → Stage 1 @ 200k│  │ └─────────────────────────────────────────────┘ │ │
│  │  → Stage 2 @ 800k│                                                     │ │
│  └──────────────────┘  ┌─────────────────────────────────────────────────┐ │
│                         │ Action Loop Detector                            │ │
│                         │ ┌─────────────────────────────────────────────┐ │ │
│                         │ │ ⚠️ ACTION LOOP DETECTED!                   │ │ │
│                         │ │ Environment: 45                            │ │ │
│                         │ │ Action: PRESERVE (count: 8/5)              │ │ │
│                         │ └─────────────────────────────────────────────┘ │ │
│                         │                                                 │ │
│                         │ ┌─────────────────────────────────────────────┐ │ │
│                         │ │ Return Statistics                            │ │ │
│                         │ │ Avg: 125.4 | Med: 118.2 | Max: 156.7        │ │ │
│                         │ │ Histogram: [pyqtgraph bar chart]            │ │ │
│                         │ └─────────────────────────────────────────────┘ │ │
│                         └─────────────────────────────────────────────────┘ │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Status Bar: Steps: 100,000 / 1,000,000  Time: 12m 34s  GPU: 78%           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔐 Headless Mode Detection

```python
# train_ui/worker.py

def run_train(..., headless=False):
    if headless:
        # No queues created, training runs in background
        _train_headless(cfg)
        return None, None
    
    # Standard mode with bidirectional communication
    metrics_queue = multiprocessing.Queue()
    command_queue = multiprocessing.Queue()
    
    p = Process(target=_worker_with_ui, args=(cfg, metrics_queue, command_queue))
    p.start()
    
    return p, metrics_queue, command_queue
```

## 📦 Данные между процессами

### Performance constraints:
- ProgressMsg sent every **1000 steps** (default n_steps=2048)
- Command processing is **asynchronous** and non-blocking
- UI updates happen in **separate thread** to avoid freezing
- Memory overhead: ~50KB per message (well within limits)

### Error handling:
- All `progress_cb` calls wrapped in try/except
- Corrupt messages logged but don't crash worker
- Queue overflow handled with bounded queues (size=1000)
