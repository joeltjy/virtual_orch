#pragma once

class WorkspaceCanvas;

namespace views {

struct PerformanceView {
    static auto applyTo(WorkspaceCanvas &canvas) -> void;
};

} // namespace views
