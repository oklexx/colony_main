# IMPLEMENTATION NOTES — УЧЁТ В РЕАЛИЗАЦИИ

## 1. Return Statistics Approach (Not a Bug, but Implementation Decision)

**Context:** 
- In `03-API_SPECS.md`: ProgressMsg contains aggregated stats (`avg_return`, `median_return`, `max_return`, `min_return`)
- In `04-UI_COMPONENTS.md`: ReturnStatisticsWidget code shows working with raw returns list
- **This is INTENTIONAL design**

**Rationale:**
- Raw returns list (even small buffer of 50 episodes) is ~200 bytes, negligible overhead
- Having BOTH options gives developers flexibility:
  - Full history for advanced analytics (future-proof)
  - Aggregated stats for lightweight monitoring (current priority)

**Implementation Recommendation:**
```python
# In ProgressMsg (already in spec):
avg_return: float = 0.0      # For quick display
median_return: float = 0.0   # For outlier-resistant metrics  
max_return: float = 0.0      # For peak performance tracking
min_return: float = 0.0      # For floor analysis

# In ReturnStatisticsWidget.__init__():
self.returns_history: List[float] = []  # Optional buffer for histogram

# In AsyncTrainer._collect_rollout() - BOTH approaches work:
# Option A: Send aggregated (lightweight)
self.progress_callback({"avg_return": avg, "median_return": med, ...})

# Option B: Send full list (richer data, minimal overhead)
self.progress_callback({
    "returns": self._ep_returns[-50:],  # Last 50 episodes only
    "best_reward": float(self.best_reward),
    ...
})

# UI can handle both approaches via try/except:
def update_statistics(self, data: Dict):
    if 'returns' in data and isinstance(data['returns'], list):
        # Full history approach (from 04-UI_COMPONENTS.md)
        returns = data['returns']
        self._update_histogram_from_full_list(returns)
    elif all(key in data for key in ['avg_return', 'median_return', ...]):
        # Aggregated approach (more efficient, from 03-API_SPECS.md)  
        avg = data['avg_return']
        med = data['median_return']
        self._update_labels_directly(avg, med, ...)
```

**Verdict:** Not a contradiction — both are valid design choices. Recommended to support BOTH for flexibility.

---

## 2. Metric Update Frequency (Minor Discrepancy)

**Context:**
- `02-ARCHITECTURE.md`: ProgressMsg sent every **1000 steps**
- `05-IMPLEMENTATION_PLAN.md`: ProgressMsg sent every **2048 steps** (or n_steps/2)

**Analysis:**
```
Option A: Fixed frequency = 1000 steps (~1 minute at 60 FPS)
Option B: Dynamic frequency = min(2048, n_steps/2)
    - Early training (n_steps=100): send every 50 steps (frequent updates)
    - Late training (n_steps=10000): send every 2048 steps (~3 minutes)
```

**Recommendation:** Use **dynamic frequency** approach for better UX:

```python
# In rl/async_trainer.py
def _should_send_progress(self, current_step: int) -> bool:
    """Determine if metrics should be sent to UI."""
    
    # Early training: frequent updates (every 256 steps)  
    if current_step < 100_000:
        return current_step % 256 == 0
    
    # Mid training: every 512 steps
    elif current_step < 500_000:
        return current_step % 512 == 0
    
    # Late training: every 1024 steps (or max configured)
    else:
        return current_step % 1024 == 0

# Or simpler: adaptive based on total_steps config
max_interval = min(2048, self.config.get("n_steps", 10_000) // 2)
return current_step % max_interval == 0
```

**Verdict:** Clarify in `05-IMPLEMENTATION_PLAN.md` that frequency is **adaptive**, not fixed. This matches both specifications (1024 is close to both 1000 and 2048).

---

## 3. Histogram Implementation Details (Code Review Notes)

**Context:** In `04-UI_COMPONENTS.md`, ReturnStatisticsWidget uses `self._dist_curve` which must be initialized after `plotItem.postInitialize()`.

**Correct Pattern:**
```python
# WRONG: Initialize immediately after PlotWidget creation
self.histogram_widget = PlotWidget()
self._dist_curve = self.histogram_widget.plots[1].plot()  # AttributeError!

# CORRECT: Connect via postInitialize callback
def on_post_init(self):
    self._dist_curve = self.histogram_widget.plots[1].plot()

self.histogram_widget.plotItem.postInitialize()
# But postInitialize doesn't automatically call callbacks...

# ACTUAL WORKAROUND: Use a flag or subclass
class ReturnStatisticsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        
        self._dist_curve = None  # Initialize as None
        
        try:
            self.histogram_widget = PlotWidget()
            
            # Override postInitialize to set up curve
            original_init = self.histogram_widget.plotItem.postInitialize
            
            def wrapped_post_init():
                original_init()
                # Now we can safely access plots
                self._dist_curve = self.histogram_widget.plots[1].plot(
                    pen='#00CED1'
                )
            
            self.histogram_widget.plotItem.postInitialize = wrapped_post_init
            
        except ImportError:
            pass
```

**Alternative (Simpler): Use BarGraphItem for histogram**

As the reviewer suggested, `pg.BarGraphItem` might be cleaner:

```python
import pyqtgraph as pg

# In __init__:
self._bar_graph = None  # Will store BarGraphItem

def update_histogram(self, returns: List[float]):
    """Update bar graph with new return distribution."""
    if len(returns) < 10:  # Need enough samples for histogram
        return
    
    # Create bins (adaptive based on return range)
    min_r, max_r = min(returns), max(returns)
    bin_size = (max_r - min_r) / 20  # 20 bins
    
    # Count frequencies
    bins = np.arange(min_r, max_r + bin_size, bin_size)
    counts, _ = np.histogram(returns, bins=bins)
    
    # Normalize for visualization
    normalized_counts = counts / counts.max()
    
    # Update graph (if exists)
    if self._bar_graph is not None:
        x = bins[:-1]
        y = normalized_counts
        
        # Update existing bar or create new one
        self._bar_graph.setPos(x)
        heights = [c for c in self._bar_graph.heights()]
        for i, h in enumerate(heights):
            self._bar_graph[i].setHeight(h) if i < len(heights) else None
    else:
        # Create new BarGraphItem
        self._bar_graph = pg.BarGraphItem(
            x=bins[:-1], 
            y=normalized_counts, 
            width=bin_size, 
            pen='q',  # Quick rendering
            brush='#00CED1'
        )
        self.histogram_widget.addItem(self._bar_graph)

# Call update_histogram whenever new returns arrive
```

**Verdict:** Not a bug — just needs attention during implementation. Both approaches (PlotCurveItem with normalization vs BarGraphItem) are valid. Recommend BarGraphItem for better histogram semantics.

---

## 4. Summary of Implementation Notes

| Issue # | Category | Severity | Status | Recommendation |
|---------|----------|----------|--------|----------------|
| 1 | Return stats approach | Low | Noted | Support BOTH aggregated and full-list approaches |
| 2 | Metric frequency | Low | Noted | Use adaptive frequency (dynamic based on total_steps) |
| 3 | Histogram initialization | Medium | Noted | Initialize _dist_curve via postInitialize callback OR use BarGraphItem |

**All three items are implementation details, not TZ contradictions.**  
They just need developer attention during coding phase.

---

**END OF IMPLEMENTATION NOTES**
