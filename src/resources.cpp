#include "colony/resources.h"

namespace colony {

Sunduk Sunduk::from_dict(const std::unordered_map<std::string, int64_t>& d) {
    static const char* names[SUNDUK_SIZE] = {
        "gold", "food", "coal", "iron", "oil", "stone", "water", "wood", "energy"};
    Sunduk s;
    for (int i = 0; i < SUNDUK_SIZE; i++) {
        auto it = d.find(names[i]);
        s.items_[i] = (it != d.end()) ? it->second : 0;
    }
    return s;
}

}  // namespace colony