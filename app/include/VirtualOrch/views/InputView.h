#pragma once

class WorkspaceCanvas;

namespace views {

struct InputView {
    static auto applyTo(WorkspaceCanvas &canvas) -> void;
};

} // namespace views
