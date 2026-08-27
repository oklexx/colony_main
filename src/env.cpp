#include "colony/env.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#include <json.hpp>

namespace colony {

namespace {

const char* SEASON_NAMES_ENV[4] = {"spring", "summer", "autumn", "winter"};

// Стили курикулума (аналог CURRICULUM_STAGES в rl/env.py)
const char* CURRICULUM_STAGE_1[] = {
    "House", "SmallHouse", "Farm", "Garden", "Mushroom", "WaterChannel",
    "Refinery", "Fish", "HuntingLand", "CowFarm", "Apiary", "Hothouse",
    "Puerperal", "Road"};
const char* CURRICULUM_STAGE_2[] = {
    "Sawmill", "Coalmine", "CoalCut", "Ironmine", "PowerStation",
    "HydroStation", "AirStation", "Torchlight", "Goldmine", "BigHouse",
    "BigFarm"};
const char* CURRICULUM_STAGE_3[] = {
    "BigSawmill", "BigRefinary", "BigIronmine", "WaterMill",
    "SmallAtomStation", "AtomStation", "SuperHouse"};

}  // namespace

ColonyEnvCpp::ColonyEnvCpp(const std::vector<BaseData>& base_data,
                           const std::vector<BaseEvent>& events_data,
                           int64_t seed, int map_size, int curriculum_stage,
                           const std::vector<std::string>& unlock_ids,
                           const RewardConfig& cfg)
    : base_data_(std::make_shared<const std::vector<BaseData>>(base_data)),
      events_data_(std::make_shared<const std::vector<BaseEvent>>(events_data)),
      cfg_(cfg),
      map_size_(map_size),
      curriculum_stage_(curriculum_stage),
      has_unlocked_(false),
      game_(*base_data_, *events_data_, seed, map_size) {
    // пул построек
    std::unordered_set<std::string> subset;
    for (const char* id : BUILD_SUBSET) subset.insert(id);
    for (const BaseData& d : *base_data_) {
        if (d.id != DEPOT_ID && subset.count(d.id)) {
            build_ids_.push_back(d.id);
            build_data_.push_back(&d);
        }
    }
    n_build_ = (int)build_ids_.size();
    for (int i = 0; i < n_build_; i++) build_id_to_idx_[build_ids_[i]] = i;
    manager_base_ = A_BUILD0 + n_build_;

    // курикулум
    if (curriculum_stage > 0) {
        for (int s = 1; s <= curriculum_stage && s <= 3; s++) {
            const char* const* list = nullptr;
            int n = 0;
            if (s == 1) { list = CURRICULUM_STAGE_1; n = (int)(sizeof(CURRICULUM_STAGE_1) / sizeof(char*)); }
            if (s == 2) { list = CURRICULUM_STAGE_2; n = (int)(sizeof(CURRICULUM_STAGE_2) / sizeof(char*)); }
            if (s == 3) { list = CURRICULUM_STAGE_3; n = (int)(sizeof(CURRICULUM_STAGE_3) / sizeof(char*)); }
            for (int i = 0; i < n; i++) unlocked_.insert(list[i]);
        }
        has_unlocked_ = true;
    }
    if (!unlock_ids.empty()) {
        for (const std::string& id : unlock_ids) unlocked_.insert(id);
        has_unlocked_ = true;
    }

    compute_catalog();
    // RL-среда не использует undo — отключаем для производительности
    game_.set_enable_undo(false);
}

void ColonyEnvCpp::set_curriculum_stage(int stage) {
    curriculum_stage_ = stage;
    unlocked_.clear();
    has_unlocked_ = false;
    if (stage > 0) {
        for (int s = 1; s <= stage && s <= 3; s++) {
            const char* const* list = nullptr;
            int n = 0;
            if (s == 1) { list = CURRICULUM_STAGE_1; n = (int)(sizeof(CURRICULUM_STAGE_1) / sizeof(char*)); }
            if (s == 2) { list = CURRICULUM_STAGE_2; n = (int)(sizeof(CURRICULUM_STAGE_2) / sizeof(char*)); }
            if (s == 3) { list = CURRICULUM_STAGE_3; n = (int)(sizeof(CURRICULUM_STAGE_3) / sizeof(char*)); }
            for (int i = 0; i < n; i++) unlocked_.insert(list[i]);
        }
        has_unlocked_ = true;
    }
    compute_catalog();
}

// Map Python season index (0=spring, 1=summer, 2=autumn, 3=winter) to C++ Season enum
    static Season python_season_to_cpp(int py_idx) {
        switch (py_idx) {
            case 0: return SEASON_SPRING;
            case 1: return SEASON_SUMMER;
            case 2: return SEASON_AUTUMN;
            case 3: return SEASON_WINTER;
            default: return SEASON_SPRING;
        }
    }

    void ColonyEnvCpp::compute_catalog() {
        sale_prices_.resize(SUNDUK_SIZE);
        for (int i = 0; i < SUNDUK_SIZE; i++)
            sale_prices_[i] = (float)((double)SALE_SUNDUK[i] / 1000.0);

        std::vector<std::vector<float>> catalog(n_build_, std::vector<float>(3, 0.0f));
        std::vector<std::vector<float>> season_mask(4, std::vector<float>(n_build_, 0.0f));
        for (int i = 0; i < n_build_; i++) {
            const BaseData& d = *build_data_[i];
            int64_t net = 0;
            for (int j = 0; j < SUNDUK_SIZE; j++)
                net += (d.profit[j] - d.consume[j]) * SALE_SUNDUK[j];
            int64_t season_days = 0;
            for (int s = 0; s < 4; s++)
                if (d.work_seasons[python_season_to_cpp(s)]) season_days += SEASON_DAYS[python_season_to_cpp(s)];
            catalog[i][0] = (float)std::min(10.0, (double)net * (double)season_days / 1e6);
            catalog[i][1] = (float)(std::log10((double)std::max<int64_t>(1, d.price)) / 10.0);
            catalog[i][2] = (float)((double)d.need_workers / 10.0);
            for (int s = 0; s < 4; s++)
                if (d.work_seasons[python_season_to_cpp(s)]) season_mask[s][i] = 1.0f;
        }
        catalog_by_season_.resize(4);
        for (int s = 0; s < 4; s++) {
            catalog_by_season_[s].resize((size_t)n_build_ * 4);
            for (int i = 0; i < n_build_; i++) {
                catalog_by_season_[s][i * 4 + 0] = catalog[i][0];
                catalog_by_season_[s][i * 4 + 1] = catalog[i][1];
                catalog_by_season_[s][i * 4 + 2] = catalog[i][2];
                catalog_by_season_[s][i * 4 + 3] = season_mask[s][i];
            }
        }
    }

void ColonyEnvCpp::reset(int64_t seed) {
    game_ = Game(*base_data_, *events_data_, seed, map_size_);
    steps_ = 0;
    ep_return_ = 0.0;
    last_reward_ = 0.0;
    last_daily_value_ = 0.0;
    last_chain_daily_ = 0.0;
    tax_due_days_ = 0;
    produced_.clear();
    building_before_.clear();
    chain_done_.clear();
    first_working_.clear();
}

std::vector<Season> ColonyEnvCpp::step_seasons(int y, int m, int d, int n_days) const {
    std::vector<Season> seasons;
    if (n_days == 1) {
        seasons.push_back(season_for_month(m));
        return seasons;
    }
    for (int i = 0; i < n_days; i++) {
        seasons.push_back(season_for_month(m));
        d += 1;
        if (d > days_in_month(y, m)) {
            d = 1;
            m += 1;
            if (m > 12) {
                m = 1;
                y += 1;
            }
        }
    }
    return seasons;
}

int ColonyEnvCpp::road_count() const {
    int n = 0;
    for (const Base& b : game_.bases)
        if (b.data->id == ROAD_ID) n++;
    return n;
}

bool ColonyEnvCpp::lot_ok(int x, int y, int need_earth, bool no_near_base) const {
    const Game& g = game_;
    if (g.base_in_box(x, y) != nullptr) return false;
    int8_t cur = g.earth.lot(x, y);
    if (!(cur >= LT_NORMAL && cur < LT_LAST)) return false;
    if (need_earth != LT_EVERYWHERE && cur != need_earth) return false;
    if (no_near_base) {
        const int dx4[4] = {1, -1, 0, 0};
        const int dy4[4] = {0, 0, 1, -1};
        for (int i = 0; i < 4; i++) {
            if (g.earth.in_bounds(x + dx4[i], y + dy4[i]) &&
                g.base_in_box(x + dx4[i], y + dy4[i]) != nullptr)
                return false;
        }
    }
    return true;
}

std::optional<std::pair<int, int>> ColonyEnvCpp::find_lot(int need_earth, bool no_near_base) {
    const Game& g = game_;
    const int ms = g.map_size();
    const int cx = g.earth.init_sel_x, cy = g.earth.init_sel_y;
    // Определяем максимальный радиус, ограниченный границами карты
    int max_r = LOT_RADIUS;
    if (cx < max_r) max_r = cx;
    if (cx >= ms - max_r) max_r = ms - 1 - cx;
    if (cy < max_r) max_r = cy;
    if (cy >= ms - max_r) max_r = ms - 1 - cy;
    // Ограничиваем также LOT_RADIUS
    if (max_r > LOT_RADIUS) max_r = LOT_RADIUS;

    const int8_t* lots = g.earth.lots().data();
    const int32_t* idx_map = g.base_index_map().data();

    for (int r = 0; r <= max_r; r++) {
        // Верхняя и нижняя границы кольца
        {
            int y = cy - r;
            if (y >= 0) {
                int x_start = (cx - r >= 0) ? cx - r : 0;
                int x_end = (cx + r < ms) ? cx + r : ms - 1;
                for (int x = x_start; x <= x_end; x++) {
                    size_t idx = (size_t)y * ms + x;
                    if (idx_map[idx] < 0) {
                        int8_t cur = lots[idx];
                        if (cur >= LT_NORMAL && cur < LT_LAST &&
                            (need_earth == LT_EVERYWHERE || cur == need_earth) &&
                            !no_near_base)
                        {
                            return std::make_pair(x, y);
                        }
                        if (cur >= LT_NORMAL && cur < LT_LAST &&
                            (need_earth == LT_EVERYWHERE || cur == need_earth) &&
                            no_near_base) {
                            // Проверяем соседей
                            bool ok = true;
                            if (x > 0 && idx_map[idx - 1] >= 0) ok = false;
                            else if (x < ms - 1 && idx_map[idx + 1] >= 0) ok = false;
                            else if (y > 0 && idx_map[idx - ms] >= 0) ok = false;
                            else if (y < ms - 1 && idx_map[idx + ms] >= 0) ok = false;
                            if (ok) return std::make_pair(x, y);
                        }
                    }
                }
            }
        }
        {
            int y = cy + r;
            if (y < ms) {
                int x_start = (cx - r >= 0) ? cx - r : 0;
                int x_end = (cx + r < ms) ? cx + r : ms - 1;
                for (int x = x_start; x <= x_end; x++) {
                    size_t idx = (size_t)y * ms + x;
                    if (idx_map[idx] < 0) {
                        int8_t cur = lots[idx];
                        if (cur >= LT_NORMAL && cur < LT_LAST &&
                            (need_earth == LT_EVERYWHERE || cur == need_earth) &&
                            !no_near_base)
                        {
                            return std::make_pair(x, y);
                        }
                        if (cur >= LT_NORMAL && cur < LT_LAST &&
                            (need_earth == LT_EVERYWHERE || cur == need_earth) &&
                            no_near_base) {
                            bool ok = true;
                            if (x > 0 && idx_map[idx - 1] >= 0) ok = false;
                            else if (x < ms - 1 && idx_map[idx + 1] >= 0) ok = false;
                            else if (y > 0 && idx_map[idx - ms] >= 0) ok = false;
                            else if (y < ms - 1 && idx_map[idx + ms] >= 0) ok = false;
                            if (ok) return std::make_pair(x, y);
                        }
                    }
                }
            }
        }
        // Левая и правая границы кольца (без углов, чтобы не дублировать)
        if (r > 0) {
            int x_left = cx - r;
            if (x_left >= 0) {
                int y_start = cy - r + 1;
                int y_end = cy + r - 1;
                for (int y = y_start; y <= y_end; y++) {
                    size_t idx = (size_t)y * ms + x_left;
                    if (idx_map[idx] < 0) {
                        int8_t cur = lots[idx];
                        if (cur >= LT_NORMAL && cur < LT_LAST &&
                            (need_earth == LT_EVERYWHERE || cur == need_earth) &&
                            !no_near_base)
                        {
                            return std::make_pair(x_left, y);
                        }
                        if (cur >= LT_NORMAL && cur < LT_LAST &&
                            (need_earth == LT_EVERYWHERE || cur == need_earth) &&
                            no_near_base) {
                            bool ok = true;
                            if (x_left > 0 && idx_map[idx - 1] >= 0) ok = false;
                            else if (x_left < ms - 1 && idx_map[idx + 1] >= 0) ok = false;
                            else if (y > 0 && idx_map[idx - ms] >= 0) ok = false;
                            else if (y < ms - 1 && idx_map[idx + ms] >= 0) ok = false;
                            if (ok) return std::make_pair(x_left, y);
                        }
                    }
                }
            }
            int x_right = cx + r;
            if (x_right < ms) {
                int y_start = cy - r + 1;
                int y_end = cy + r - 1;
                for (int y = y_start; y <= y_end; y++) {
                    size_t idx = (size_t)y * ms + x_right;
                    if (idx_map[idx] < 0) {
                        int8_t cur = lots[idx];
                        if (cur >= LT_NORMAL && cur < LT_LAST &&
                            (need_earth == LT_EVERYWHERE || cur == need_earth) &&
                            !no_near_base)
                        {
                            return std::make_pair(x_right, y);
                        }
                        if (cur >= LT_NORMAL && cur < LT_LAST &&
                            (need_earth == LT_EVERYWHERE || cur == need_earth) &&
                            no_near_base) {
                            bool ok = true;
                            if (x_right > 0 && idx_map[idx - 1] >= 0) ok = false;
                            else if (x_right < ms - 1 && idx_map[idx + 1] >= 0) ok = false;
                            else if (y > 0 && idx_map[idx - ms] >= 0) ok = false;
                            else if (y < ms - 1 && idx_map[idx + ms] >= 0) ok = false;
                            if (ok) return std::make_pair(x_right, y);
                        }
                    }
                }
            }
        }
    }
    return std::nullopt;
}

double ColonyEnvCpp::year_production_value(const BaseData& d) const {
    int64_t v = 0;
    for (int i = 0; i < SUNDUK_SIZE; i++) v += d.profit[i] * SALE_SUNDUK[i];
    int64_t season_days = 0;
    for (int s = 0; s < 4; s++)
        if (d.work_seasons[s]) season_days += SEASON_DAYS[s];
    return (double)(v * season_days);
}

double ColonyEnvCpp::net_worth() const {
    if (net_worth_valid_) return cached_net_worth_;
    double v = (double)game_.money - (double)game_.credit;
    for (int i = 0; i < SUNDUK_SIZE; i++)
        v += (double)(game_.sunduk[i] * SALE_SUNDUK[i]);
    for (const Base& b : game_.bases) {
        const BaseData& d = *b.data;
        if (d.live_years) {
            double f = (double)b.live_time / (double)d.live_time_total();
            v += (double)d.price * (0.25 + 0.75 * f);
        } else {
            v += (double)d.price;
        }
    }
    cached_net_worth_ = v;
    net_worth_valid_ = true;
    return v;
}

double ColonyEnvCpp::net_worth(const Game& g) const {
    double v = (double)g.money - (double)g.credit;
    for (int i = 0; i < SUNDUK_SIZE; i++)
        v += (double)(g.sunduk[i] * SALE_SUNDUK[i]);
    for (const Base& b : g.bases) {
        const BaseData& d = *b.data;
        if (d.live_years) {
            double f = (double)b.live_time / (double)d.live_time_total();
            v += (double)d.price * (0.25 + 0.75 * f);
        } else {
            v += (double)d.price;
        }
    }
    return v;
}

std::vector<float> ColonyEnvCpp::obs() const { return obs(game_); }

std::vector<float> ColonyEnvCpp::obs(const Game& g) const {
    if (obs_buf_.size() != (size_t)obs_size())
        obs_buf_.resize((size_t)obs_size());
    size_t idx = 0;
    auto push = [&](float v) { obs_buf_[idx++] = v; };

    std::vector<float> counts(n_build_, 0.0f);
    double sum_live = 0.0, min_live = 0.0;
    int n_building = 0, n_preserved = 0, n_worn = 0;
    int n_idle_sunduk = 0, n_idle_workers = 0;
    bool first = true;
    for (const Base& b : g.bases) {
        auto it = build_id_to_idx_.find(b.data->id);
        if (it != build_id_to_idx_.end()) counts[it->second] += 1.0f;
        if (b.data->live_years) {
            sum_live += (double)b.live_time;
            double lt = (double)b.live_time / (double)b.data->live_time_total();
            if (first || lt < min_live) min_live = lt;
            first = false;
            if (lt < 1.0) n_worn += 1;
        }
        if (b.build_days > 0) n_building += 1;
        if (b.preserved) n_preserved += 1;
        if (b.need_sunduk) n_idle_sunduk += 1;
        if (b.need_workers) n_idle_workers += 1;
    }
    const Sunduk& a = g.sunduk;
    int season_idx;
    switch (g.season) {
        case SEASON_SPRING: season_idx = 0; break;
        case SEASON_SUMMER: season_idx = 1; break;
        case SEASON_AUTUMN: season_idx = 2; break;
        case SEASON_WINTER: season_idx = 3; break;
        default: season_idx = 0;
    }

    push((float)((double)(g.year - START_YEAR) / 50.0));
    push((float)((double)g.month / 12.0));
    push((float)((double)g.day / 31.0));
    push((float)((double)season_idx / 3.0));
    push((float)((double)g.money / 2e5));
    push((float)((double)g.credit / 2e5));
    push((float)((double)g.people / 100.0));
    push((float)((double)g.busy_people / 100.0));
    for (int i = 0; i < SUNDUK_SIZE; i++)
        push((float)std::min(1.0, (double)a[i] / 100.0));
    push((float)((double)g.now_home_places() / 100.0));
    push((float)((double)g.now_need_workers() / 100.0));
    push((float)((double)g.free_people() / 100.0));
    push((float)(std::max(0.0, (double)(g.people - g.now_home_places())) / 100.0));
    push((float)(g.annual_tax_due() ? 1.0 : 0.0));
    push((float)(g.main_tax_due() ? 1.0 : 0.0));
    push((float)((double)g.annual_tax_amount() / 2e4));
    push((float)((double)g.main_tax_amount() / 5e5));
    push((float)((double)g.days_alive / 3650.0));
    push((float)((double)curriculum_stage_ / 3.0));
    for (float c : counts) push((float)((double)c / 10.0));
    push((float)(sum_live / 5000.0));
    push((float)min_live);
    push((float)((double)n_building / 10.0));
    push((float)((double)n_preserved / 10.0));
    push((float)((double)n_worn / 10.0));
    push((float)((double)n_idle_sunduk / 10.0));
    push((float)((double)n_idle_workers / 10.0));
    for (float p : sale_prices_) push(p);
    const std::vector<float>& cat = catalog_by_season_[season_idx];
    for (float c : cat) push(c);
    return obs_buf_;
}

ColonyEnvCpp::StepOut ColonyEnvCpp::step(int action) {
    Game& g = game_;
    double rew = 0.0;

    // Capture ALL base uids before action for correct built_price calculation
    std::unordered_set<int64_t> uids_before;
    for (const Base& b : g.bases) uids_before.insert(b.uid);

    // налог платится автоматически
    if (g.annual_tax_due()) {
        if (g.money >= g.annual_tax_amount()) {
            g.pay_annual_tax();
            rew += 15.0;
        } else {
            rew -= 5.0;
        }
    } else if (g.main_tax_due()) {
        if (g.money >= g.main_tax_amount()) {
            g.pay_main_tax();
            rew += 30.0;
            rew += cfg_.tax_bonus;
        } else {
            rew -= 5.0;
        }
    }
    double net0 = cfg_.disable_net_worth ? 0.0 : net_worth();
    int y0 = g.year, m0 = g.month, d0 = g.day;

    // Track bases under construction before action (for first-production bonus)
    std::unordered_set<int64_t> building_before;
    for (const Base& b : g.bases)
        if (b.build_days > 0) building_before.insert(b.uid);
    building_before_ = building_before;

    // действия
    if (action == A_DAY || action == A_WEEK) {
        // ничего
    } else if (action >= A_BUILD0 && action < A_BUILD0 + n_build_) {
        const BaseData* d = build_data_[action - A_BUILD0];
        std::optional<std::pair<int, int>> cell;
        if (has_unlocked_ && !unlocked_.count(d->id)) {
            rew -= 1.0;
        } else if (d->id == ROAD_ID && road_count() >= MAX_ROADS) {
            rew -= 1.0;
        } else {
            cell = find_lot(d->need_earth, d->no_near_base);
        }
        if (!cell || !g.build(d->id, cell->first, cell->second).first)
            rew -= 1.0;
        else {
            rew += cfg_.build_bonus * year_production_value(*d);
            invalidate_net_worth();
        }
    } else if (action == manager_base_ + 0) {  // Улучшить землю
        auto cell2 = find_lot(LT_EVERYWHERE, false);
        if (!cell2 || !g.good_earth(cell2->first, cell2->second).first)
            rew -= 1.0;
    } else if (action == manager_base_ + 1) {  // Ремонт
        Base* b = g.find_slowest_base();
        if (b == nullptr || !g.restore(b->x, b->y).ok) rew -= 1.0;
        else invalidate_net_worth();
    } else if (action == manager_base_ + 2) {  // Ремонт всех
        if (!g.restore_all().ok) rew -= 1.0;
        else invalidate_net_worth();
    } else if (action == manager_base_ + 3) {  // Снос
        Base* b = g.find_slowest_base();
        if (b == nullptr || b->data->id == DEPOT_ID ||
            !g.destroy(b->x, b->y).first)
            rew -= 1.0;
        else invalidate_net_worth();
    } else if (action == manager_base_ + 4) {  // Консервация
        const Base* best = nullptr;
        for (const Base& b : g.bases) {
            if (!b.preserved && !b.data->no_preserve && b.data->id != DEPOT_ID) {
                if (best == nullptr || b.live_time < best->live_time) best = &b;
            }
        }
        if (best == nullptr || !g.preserve(best->x, best->y).first)
            rew -= 1.0;
        else {
            rew -= 0.5;
            invalidate_net_worth();
        }
    } else if (action == manager_base_ + 5) {  // Разконсервация
        const Base* first_preserved = nullptr;
        for (const Base& b : g.bases) {
            if (b.preserved) {
                first_preserved = &b;
                break;
            }
        }
        if (first_preserved == nullptr ||
            !g.preserve(first_preserved->x, first_preserved->y).first)
            rew -= 1.0;
        else {
            rew -= 0.5;
            invalidate_net_worth();
        }
    } else if (action == manager_base_ + 6) {  // Продать излишки
        Sunduk counts;
        for (int i = 0; i < SUNDUK_SIZE; i++)
            counts[i] = std::max<int64_t>(0, g.sunduk[i] - 200);
        auto r = g.market_sell(counts);
        if (!r.ok || r.total <= 0)
            rew -= 1.0;
        else
            rew += cfg_.sale_bonus * (double)r.total;
    } else if (action == manager_base_ + 7) {  // Купить еду
        int64_t need = std::max<int64_t>(0, 400 - g.sunduk[FOOD]);
        if (need > 0) {
            Sunduk counts;
            counts[FOOD] = need;
            if (!g.market_buy(counts).ok) rew -= 1.0;
        } else {
            rew -= 0.1;
        }
    } else if (action == manager_base_ + 8) {  // Взять кредит
        if (g.credit >= 100000 || !g.bank_take(50000).first) rew -= 1.0;
    } else if (action == manager_base_ + 9) {  // Вернуть кредит
        int64_t give = std::min<int64_t>(50000, g.credit);
        if (g.credit <= 0 || !g.bank_give(give).first) rew -= 1.0;
    } else if (action == manager_base_ + 10) {  // Заплатить налог
        rew -= 0.5;
    }

    // прожить день/неделю
    std::vector<DayResult> results;
    if (action == A_WEEK) {
        results = g.advance_week();
    } else {
        auto r = g.advance_day();
        if (r.ok) results.push_back(r.res);
    }
    if (results.empty() && (action == A_DAY || action == A_WEEK)) rew -= 1.0;
    invalidate_net_worth();

    int64_t built_price = 0;
    for (const Base& b : g.bases) {
        if (uids_before.find(b.uid) == uids_before.end()) {
            built_price += b.data->price;
        }
    }
    if (!cfg_.disable_net_worth)
        rew += 0.005 * (net_worth() - net0 - (double)built_price);
    for (const DayResult& r : results) {
        rew -= 0.005 * (double)g.credit / 1000.0;
        rew += (double)r.born * 2.0 + (double)r.people_arrived * 2.0;
        rew -= (double)r.died * 30.0;
        rew -= (double)r.base_lost * 50.0;
        if (r.home_overflow) rew -= 2.0;
    }

    // цепочки и ежедневный доход
    std::vector<Season> seasons = step_seasons(y0, m0, d0, (int)results.size());
    std::vector<int64_t> worked_any;
    double daily_total = 0.0, chain_daily = 0.0;
    for (Season season : seasons) {
        std::vector<std::pair<int, int>> day_working;
        std::vector<std::vector<std::pair<int, int>>> producers(SUNDUK_SIZE);
        for (const Base& b : g.bases) {
            if (b.build_days == 0 && !b.preserved && b.state_empty() &&
                b.data->season_works(season)) {
                auto pos = std::make_pair(b.x, b.y);
                day_working.push_back(pos);
                worked_any.push_back(b.uid);
                const BaseData& d = *b.data;
                for (int j = 0; j < SUNDUK_SIZE; j++) {
                    daily_total +=
                        (double)((d.profit[j] - d.consume[j]) * SALE_SUNDUK[j]);
                    if (d.profit[j] > 0) producers[j].push_back(pos);
                }
            }
        }
        for (const auto& pos : day_working) {
            // Find base by position
            const Base* b = g.base_in_box(pos.first, pos.second);
            if (!b) continue;
            const BaseData& d = *b->data;
            for (int i = 0; i < SUNDUK_SIZE; i++) {
                if (d.consume[i] <= 0 || producers[i].empty()) continue;
                auto key = std::make_pair(d.id, i);
                if (!chain_done_.count(key)) {
                    chain_done_.insert(key);
                    rew += cfg_.chain_bonus * year_production_value(d);
                }
                chain_daily += cfg_.chain_daily;
            }
        }
    }
    if (!cfg_.disable_daily_income) rew += cfg_.daily_income * daily_total;
    rew += chain_daily;
    last_daily_value_ = daily_total;
    last_chain_daily_ = chain_daily;
    for (int64_t uid : worked_any) {
        if (produced_.count(uid) || building_before.count(uid)) continue;
        const Base* b = nullptr;
        for (const Base& bb : g.bases)
            if (bb.uid == uid) { b = &bb; break; }
        if (!b) continue;
        produced_.insert(uid);
        if (!first_working_.count(b->data->id)) {
            first_working_.insert(b->data->id);
            rew += cfg_.novelty;
        }
    }

    // автоматическое обслуживание
    for (Base& b : g.bases) {
        int64_t max_live = b.data->live_time_total();
        if (b.data->live_years && (double)b.live_time < (double)max_live * 0.3) {
            int64_t price = b.data->restore_price(b.live_time);
            if (price > 0 && g.money >= price) {
                g.save_undo();
                b.live_time = max_live;
                g.money -= price;
            }
        }
    }

    // терминалы
    steps_ += 1;
    ep_return_ += rew;
    last_reward_ = rew;
    bool terminated = false, truncated = false;
    auto ov = g.game_over();
    if (g.annual_tax_due() || g.main_tax_due())
        tax_due_days_ += 1;
    else
        tax_due_days_ = 0;
    if (ov.has_value()) {
        terminated = true;
        rew -= cfg_.game_over_penalty;
    } else if (tax_due_days_ >= TAX_GRACE_DAYS) {
        terminated = true;
        rew -= cfg_.game_over_penalty;
    } else if (g.credit > g.max_credit()) {
        terminated = true;
        rew -= cfg_.game_over_penalty;
    } else if (steps_ >= MAX_STEPS) {
        truncated = true;
    }

    rew += cfg_.survival_bonus;

    StepOut out;
    out.obs = obs();
    out.rew = rew;
    out.terminated = terminated;
    out.truncated = truncated;
    out.days = g.days_alive;
    out.people = g.people;
    out.money = g.money;
    out.n_bases = (int64_t)g.bases.size();
    out.seed = (int64_t)g.earth.seed();
    out.tax_due_days = tax_due_days_;
    out.ep_return = ep_return_;
    out.steps = steps_;
    return out;
}

// ===========================================================================
// ColonyVecEnvCpp — batched vectorized environment
// ===========================================================================

ColonyVecEnvCpp::ColonyVecEnvCpp(
    const std::vector<BaseData>& base_data,
    const std::vector<BaseEvent>& events_data,
    int n_envs, int64_t base_seed, int map_size,
    int curriculum_stage,
    const std::vector<std::string>& unlock_ids,
    const RewardConfig& cfg,
    int n_threads)
    : base_data_(std::make_shared<std::vector<BaseData>>(base_data)),
      events_data_(std::make_shared<std::vector<BaseEvent>>(events_data)),
      cfg_(cfg),
      map_size_(map_size),
      curriculum_stage_(curriculum_stage),
      unlock_ids_(unlock_ids),
      n_envs_(n_envs),
      base_seed_(base_seed),
      pool_([n_threads, n_envs]() -> size_t {
          size_t hw = std::thread::hardware_concurrency();
          if (hw == 0) hw = 4;
          int desired = n_threads > 0 ? n_threads : n_envs;
          return (size_t)std::min(desired, (int)hw);
      }()) {
    // Create N envs from shared (immutable) data
    envs_.reserve(n_envs);
    for (int i = 0; i < n_envs; ++i) {
        envs_.emplace_back(*base_data_, *events_data_, base_seed + i * 10000,
                           map_size, curriculum_stage, unlock_ids, cfg);
    }
    obs_size_ = envs_[0].obs_size();
    n_actions_ = envs_[0].n_actions();

    // Pre-allocate buffers
    size_t obs_flat = (size_t)n_envs_ * obs_size_;
    obs_buffer_.resize(obs_flat, 0.0f);
    old_obs_buffer_.resize(obs_flat, 0.0f);
    rewards_.resize(n_envs_, 0.0);
    old_rew_buffer_.resize(n_envs_, 0.0);
    terminateds_.resize(n_envs_, false);
    trunceds_.resize(n_envs_, false);
    episode_return_.resize(n_envs_, 0.0);
    episode_length_.resize(n_envs_, 0);

    // Initialize RMS with correct sizes
    obs_rms_ = RunningMeanStd(obs_size_);
    rew_rms_ = RunningMeanStd(1);
}

void ColonyVecEnvCpp::do_reset(int i, int64_t seed) {
    envs_[i].reset(seed);
    std::vector<float> obs = envs_[i].obs();
    std::copy(obs.begin(), obs.end(), obs_buffer_.data() + (size_t)i * obs_size_);
    episode_return_[i] = 0.0;
    episode_length_[i] = 0;
}

void ColonyVecEnvCpp::reset_batch(const std::vector<int64_t>& seeds) {
    for (int i = 0; i < n_envs_; ++i) {
        do_reset(i, seeds[i]);
    }
    // Copy raw obs to old_obs_buffer for RMS update order
    std::copy(obs_buffer_.begin(), obs_buffer_.end(), old_obs_buffer_.begin());
    // Normalize obs for the agent (SB3 convention: reset returns normalized obs)
    if (norm_obs_) {
        obs_rms_.normalize(obs_buffer_.data(), n_envs_, obs_size_, clip_obs_);
    }
}

void ColonyVecEnvCpp::do_step(int i, int action) {
    auto out = envs_[i].step(action);
    float* obs_ptr = obs_buffer_.data() + (size_t)i * obs_size_;
    std::copy(out.obs.begin(), out.obs.end(), obs_ptr);
    rewards_[i] = out.rew;
    terminateds_[i] = out.terminated;
    trunceds_[i] = out.truncated;
    episode_return_[i] += out.rew;
    episode_length_[i] += 1;
}

void ColonyVecEnvCpp::step_async_batch(const std::vector<int>& actions) {
    std::vector<std::future<void>> futures;
    futures.reserve(n_envs_);
    for (int i = 0; i < n_envs_; ++i) {
        int action = actions[i];
        futures.push_back(pool_.submit([this, i, action]() {
            do_step(i, action);
        }));
    }
    // Wait for all tasks to complete
    for (auto& f : futures) {
        try {
            f.get();
        } catch (const std::exception&) {
            // Step failed — leave obs/reward as-is for this env
        }
    }
}

StepBatchResult ColonyVecEnvCpp::step_wait_batch() {
    // SB3 VecNormalize order:
    // 1. Update RMS with OLD obs/rewards (from previous step)
    // 2. Normalize current obs/rewards with old RMS
    // 3. Save current raw obs/rewards as old for next step

    // Save raw obs BEFORE normalization for terminal_observation
    // (use a separate buffer to avoid copying the entire obs_buffer_)
    if (raw_obs_buf_.size() != obs_buffer_.size())
        raw_obs_buf_.resize(obs_buffer_.size());
    std::copy(obs_buffer_.begin(), obs_buffer_.end(), raw_obs_buf_.begin());

    if (norm_obs_) {
        obs_rms_.update(old_obs_buffer_.data(), n_envs_, obs_size_);
        obs_rms_.normalize(obs_buffer_.data(), n_envs_, obs_size_, clip_obs_);
    }
    if (norm_reward_) {
        for (int i = 0; i < n_envs_; ++i) {
            rew_rms_.update_scalar(old_rew_buffer_[i]);
        }
        for (int i = 0; i < n_envs_; ++i) {
            rewards_[i] = rew_rms_.normalize_reward(rewards_[i], clip_reward_);
        }
    }

    // Save current raw obs/rewards as old for next step
    std::copy(raw_obs_buf_.begin(), raw_obs_buf_.end(), old_obs_buffer_.begin());
    std::copy(rewards_.begin(), rewards_.end(), old_rew_buffer_.begin());

    // Auto-reset done envs
    StepBatchResult result;
    result.obs = obs_buffer_;
    result.rewards = rewards_;
    result.terminateds = terminateds_;
    result.trunceds = trunceds_;
    result.infos.resize(n_envs_);

    for (int i = 0; i < n_envs_; ++i) {
        if (terminateds_[i] || trunceds_[i]) {
            // terminal_observation must be RAW (unnormalized), per SB3 convention
            std::vector<float> terminal_obs(raw_obs_buf_.begin() + (size_t)i * obs_size_,
                                            raw_obs_buf_.begin() + (size_t)(i + 1) * obs_size_);

            // Build info JSON
            nlohmann::json info;
            info["terminal_observation"] = terminal_obs;
            info["episode"] = {
                {"r", episode_return_[i]},
                {"l", episode_length_[i]}
            };
            result.infos[i] = info.dump();

            // Auto-reset with new seed
            int64_t new_seed = base_seed_ + (int64_t)i * 10000 +
                               envs_[i].steps() + 100000;
            do_reset(i, new_seed);

            // Update old_obs_buffer_ for the auto-reset env (raw obs for next RMS update)
            std::copy(obs_buffer_.begin() + (size_t)i * obs_size_,
                      obs_buffer_.begin() + (size_t)(i + 1) * obs_size_,
                      old_obs_buffer_.begin() + (size_t)i * obs_size_);

            // Re-normalize the new obs (fresh from reset, not from old RMS)
            if (norm_obs_) {
                float* new_obs = obs_buffer_.data() + (size_t)i * obs_size_;
                obs_rms_.normalize(new_obs, 1, obs_size_, clip_obs_);
            }
            std::copy(obs_buffer_.begin() + (size_t)i * obs_size_,
                      obs_buffer_.begin() + (size_t)(i + 1) * obs_size_,
                      result.obs.begin() + (size_t)i * obs_size_);
        } else {
            result.infos[i] = "{}";
        }
    }

    return result;
}

void ColonyVecEnvCpp::save_normalization(const std::string& path) {
    nlohmann::json j;
    j["obs_rms"] = obs_rms_.to_json();
    j["rew_rms"] = rew_rms_.to_json();
    j["norm_obs"] = norm_obs_;
    j["norm_reward"] = norm_reward_;
    j["clip_obs"] = clip_obs_;
    j["clip_reward"] = clip_reward_;
    std::ofstream f(path);
    f << j.dump(2);
}

void ColonyVecEnvCpp::load_normalization(const std::string& path) {
    std::ifstream f(path);
    nlohmann::json j;
    f >> j;
    obs_rms_.from_json(j["obs_rms"]);
    rew_rms_.from_json(j["rew_rms"]);
    norm_obs_ = j.value("norm_obs", true);
    norm_reward_ = j.value("norm_reward", true);
    clip_obs_ = j.value("clip_obs", 10.0);
    clip_reward_ = j.value("clip_reward", 10.0);
}

}  // namespace colony