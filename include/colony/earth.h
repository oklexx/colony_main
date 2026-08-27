#pragma once

#include <cstdint>
#include <vector>

#include "colony/constants.h"
#include "colony/rng.h"

namespace colony {

class Earth {
public:
    static constexpr int DEFAULT_SIZE = 280;

    Earth(uint64_t seed, int size = DEFAULT_SIZE);

    int size() const { return size_; }
    uint64_t seed() const { return seed_; }

    int8_t lot(int x, int y) const { return lots_[y * size_ + x]; }
    int8_t sub(int x, int y) const { return subtype_[y * size_ + x]; }
    bool in_bounds(int x, int y) const {
        return x >= 0 && x < size_ && y >= 0 && y < size_;
    }

    const std::vector<int8_t>& lots() const { return lots_; }
    const std::vector<int8_t>& subtype() const { return subtype_; }
    int8_t lot_at(int idx) const { return lots_[idx]; }

    int init_sel_x = DEFAULT_SIZE / 2;
    int init_sel_y = DEFAULT_SIZE / 2;

private:
    void generate();
    void carve(int cx0, int cy0, int radius, int lot_type, MtRandom& rngr);
    void carve_soft(int cx0, int cy0, int radius, int lot_type, MtRandom& rngr);

    int size_;
    uint64_t seed_;
    std::vector<int8_t> lots_;
    std::vector<int8_t> subtype_;
};

}  // namespace colony