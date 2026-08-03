#pragma once

class WorkspaceCanvas;

namespace views {

struct DefaultView {
    static auto applyTo(WorkspaceCanvas &canvas) -> void;
};

} // namespace views
