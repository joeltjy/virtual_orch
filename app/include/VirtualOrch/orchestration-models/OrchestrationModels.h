#pragma once

#include <memory>
#include <string>
#include <vector>

#include "VirtualOrch/orchestration-models/OrchestrationModel.h"

[[nodiscard]] auto orchestrationModelNames() -> std::vector<std::string>;

[[nodiscard]] auto createOrchestrationModel(const std::string &name)
    -> std::unique_ptr<OrchestrationModel>;
