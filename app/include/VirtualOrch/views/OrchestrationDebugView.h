#pragma once

class WorkspaceCanvas;

namespace views {

struct OrchestrationDebugView {
    static auto applyTo(WorkspaceCanvas &canvas) -> void;
};

} // namespace views
