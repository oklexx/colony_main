#pragma once

#include <string>
#include <vector>

#include "colony/bases.h"
#include "colony/events.h"

namespace colony {

// Загрузка статических данных из JSON (формат core/data в эталоне).
std::vector<BaseData> load_base_data(const std::string& path);
std::vector<BaseEvent> load_events(const std::string& path);

}  // namespace colony