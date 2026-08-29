#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "colony/bases.h"
#include "colony/constants.h"
#include "colony/game.h"
#include "colony/rng.h"
#include "colony/running_mean_std.h"
#include "colony/thread_pool.h"

namespace colony {

// Коэффициенты наград (аналог env-переменных COLONY_* в rl/env.py).
struct RewardConfig {
    double build_bonus = 5.0;       // бонус за постройку × year_production_value
    double chain_bonus = 0.5;       // бонус за первую цепочку потребления
    double chain_daily = 2.0;       // ежедневный бонус за активную цепочку
    double novelty = 20.0;          // бонус за первый запуск нового типа здания
    double daily_income = 0.1;      // коэффициент ежедневного дохода
    double sale_bonus = 0.1;        // бонус за продажу ресурсов
    double tax_daily_bonus = 0.77;  // ежедневный бонус при отсутствии налогового срока
    double survival_bonus = 0.0;      // + per surviving step
    double game_over_penalty = 20.0;  // - on game over (was hardcoded 20)
    bool disable_net_worth = false;     // отключить компонент чистой стоимости
    bool disable_daily_income = false;  // отключить ежедневный доход
};

// RL-среда: точная копия ColonyEnv из rl/env.py (награды и наблюдения).
class ColonyEnvCpp {
public:
    ColonyEnvCpp(const std::vector<BaseData>& base_data,
                 const std::vector<BaseEvent>& events_data, int64_t seed,
                 int map_size = 280, int curriculum_stage = 0,
                 const std::vector<std::string>& unlock_ids = {},
                 const RewardConfig& cfg = RewardConfig());

    void set_curriculum_stage(int stage);
    void set_rewards(const RewardConfig& cfg) {
        std::lock_guard lock(*cfg_mutex_);
        cfg_ = cfg;
    }
    RewardConfig reward_config() const {
        std::lock_guard lock(*cfg_mutex_);
        return cfg_;
    }

    void reset(int64_t seed);
    std::vector<float> obs() const;
    std::vector<float> obs(const Game& g) const;

    struct StepOut {
        std::vector<float> obs;
        double rew = 0.0;
        bool terminated = false, truncated = false;
        int64_t days = 0, people = 0, money = 0, n_bases = 0;
        int64_t seed = 0, tax_due_days = 0;
        bool tax_grace_expired = false;
        double ep_return = 0.0;
        int64_t steps = 0;
    };
    StepOut step(int action);

    bool tax_grace_expired() const;
    int tax_grace_days() const;

    int n_build() const { return n_build_; }
    int n_bases() const { return (int)game_.bases.size(); }
    int n_actions() const { return A_BUILD0 + n_build_ + N_MANAGERS; }
    int obs_size() const { return 27 + n_build_ + 7 + 9 + 4 * n_build_; }
    const std::vector<std::string>& build_ids() const { return build_ids_; }
    const std::vector<const BaseData*>& build_data() const { return build_data_; }
    const Game& game() const { return game_; }
    Game& game() { return game_; }
    double last_daily_value() const { return last_daily_value_; }
    double last_chain_daily() const { return last_chain_daily_; }
    int64_t steps() const { return steps_; }
    double ep_return() const { return ep_return_; }
    double last_reward() const { return last_reward_; }

    struct Stats { int64_t days, people, bases, money; };
    Stats stats() const {
        return {game_.days_alive, game_.people, (int64_t)game_.bases.size(),
                game_.money};
    }

public:
    // Debug helpers
    std::optional<std::pair<int, int>> find_lot(int need_earth, bool no_near_base);
    bool lot_ok(int x, int y, int need_earth, bool no_near_base) const;
    double debug_net_worth() const { return net_worth(game_); }

private:
    double net_worth() const;
    double net_worth(const Game& g) const;
    double year_production_value(const BaseData& d) const;
    int road_count() const;
    std::vector<Season> step_seasons(int y, int m, int d, int n_days) const;
    void compute_catalog();

    struct PairHash {
        size_t operator()(const std::pair<int, int>& p) const {
            return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(p.second);
        }
    };

    struct ChainKeyHash {
        size_t operator()(const std::pair<std::string, int>& k) const {
            return std::hash<std::string>()(k.first) ^ (size_t)k.second;
        }
    };

    std::shared_ptr<const std::vector<BaseData>> base_data_;
    std::shared_ptr<const std::vector<BaseEvent>> events_data_;
    mutable std::shared_ptr<std::mutex> cfg_mutex_ = std::make_shared<std::mutex>();
    RewardConfig cfg_;
    int map_size_;
    int curriculum_stage_;
    std::unordered_set<std::string> unlocked_;
    bool has_unlocked_;

    std::vector<std::string> build_ids_;
    std::vector<const BaseData*> build_data_;
    int n_build_;
    std::unordered_map<std::string, int> build_id_to_idx_;
    int manager_base_;  // A_BUILD0 + n_build_

    std::vector<float> sale_prices_;
    std::vector<std::vector<float>> catalog_by_season_;  // [4][n_build*4]

    Game game_;
    int64_t steps_ = 0;
    double ep_return_ = 0.0;
    double last_reward_ = 0.0;
    double last_daily_value_ = 0.0;
    double last_chain_daily_ = 0.0;
    int64_t tax_due_days_ = 0;
    int tax_grace_days_ = 0;

    std::unordered_set<int64_t> produced_;      // uid построек, получивших бонус (аналог _produced по id() в референсе)
    std::unordered_set<int64_t> building_before_;
    std::unordered_set<std::pair<std::string, int>, ChainKeyHash> chain_done_;
    std::unordered_set<std::string> first_working_;
    mutable std::pair<int, int> cell_cache_;
    mutable std::vector<float> obs_buf_;

    // Кэш net_worth
    mutable double cached_net_worth_ = 0.0;
    mutable bool net_worth_valid_ = false;
    void invalidate_net_worth() { net_worth_valid_ = false; }
};

// Batched vectorized environment — N ColonyEnvCpp instances in one process.
// Thread-pooled parallel step/reset, integrated VecNormalize, auto-reset on done.
struct StepBatchResult {
    // Flat buffers: [n_envs * obs_size]
    std::vector<float> obs;
    // [n_envs]
    std::vector<double> rewards;
    std::vector<bool> terminateds;
    std::vector<bool> trunceds;
    // Per-env info dicts (as JSON strings for pybind11)
    std::vector<std::string> infos;
};

class ColonyVecEnvCpp {
public:
    ColonyVecEnvCpp(const std::vector<BaseData>& base_data,
                    const std::vector<BaseEvent>& events_data,
                    int n_envs, int64_t base_seed, int map_size = 280,
                    int curriculum_stage = 0,
                    const std::vector<std::string>& unlock_ids = {},
                    const RewardConfig& cfg = RewardConfig(),
                    int n_threads = 0);

    void reset_batch(const std::vector<int64_t>& seeds);
    void step_async_batch(const std::vector<int>& actions);
    StepBatchResult step_wait_batch();

    void save_normalization(const std::string& path);
    void load_normalization(const std::string& path);

    int n_envs() const { return n_envs_; }
    int obs_size() const { return obs_size_; }
    int n_actions() const { return n_actions_; }
    int n_build() const { return envs_.empty() ? 0 : envs_[0].n_build(); }
    int n_bases() const { int total = 0; for (const auto& e : envs_) total += e.n_bases(); return total; }
    int64_t n_people() const { int64_t total = 0; for (const auto& e : envs_) total += e.game().people; return total; }
    double total_daily_value() const { double t = 0; for (const auto& e : envs_) t += e.last_daily_value(); return t; }
    double total_chain_daily() const { double t = 0; for (const auto& e : envs_) t += e.last_chain_daily(); return t; }
    double mean_net_worth() const { double t = 0; for (const auto& e : envs_) t += e.debug_net_worth(); return t / (double)n_envs_; }
    const float* obs_buffer() const { return obs_buffer_.data(); }

    void set_curriculum_stage(int stage) {
        curriculum_stage_ = stage;
        for (auto& env : envs_) env.set_curriculum_stage(stage);
    }
    void set_rewards(const RewardConfig& cfg) {
        cfg_ = cfg;
        for (auto& env : envs_) env.set_rewards(cfg);
    }

private:
    void do_step(int i, int action);
    void do_reset(int i, int64_t seed);
    void update_and_normalize_obs(float* raw_obs);
    double normalize_reward(double raw_rew);

    std::shared_ptr<std::vector<BaseData>> base_data_;
    std::shared_ptr<std::vector<BaseEvent>> events_data_;
    RewardConfig cfg_;
    int map_size_;
    int curriculum_stage_;
    std::vector<std::string> unlock_ids_;
    int n_envs_;
    int obs_size_;
    int n_actions_;
    int64_t base_seed_;

    std::vector<ColonyEnvCpp> envs_;
    ThreadPool pool_;

    // Pre-allocated buffers
    std::vector<float> obs_buffer_;       // [n_envs * obs_size]
    std::vector<float> old_obs_buffer_;   // for RMS update (SB3 order)
    std::vector<float> raw_obs_buf_;      // for terminal_observation (reusable)
    std::vector<double> rewards_;         // [n_envs]
    std::vector<double> old_rew_buffer_;  // for reward RMS
    std::vector<bool> terminateds_;
    std::vector<bool> trunceds_;

    // VecNormalize stats
    RunningMeanStd obs_rms_;
    RunningMeanStd rew_rms_;
    bool norm_obs_ = true;
    bool norm_reward_ = true;
    double clip_obs_ = 10.0;
    double clip_reward_ = 10.0;

    // Episode tracking per env
    std::vector<double> episode_return_;
    std::vector<int64_t> episode_length_;
};

}  // namespace colony