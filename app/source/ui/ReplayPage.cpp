#include "VirtualOrch/ui/ReplayPage.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/ProjectPaths.h"
#include "VirtualOrch/reduction/DenseTypes.h"
#include "VirtualOrch/ui/UiConstants.h"
#include "VirtualOrch/ui/WorkspacePage.h"

namespace {

const juce::Colour kInk{juce::Colours::lightgrey};

auto mono() -> juce::Font { return juce::Font(juce::Font::getDefaultMonospacedFontName(), 13.0f, juce::Font::plain); }

auto noteName(int midi) -> juce::String { return juce::MidiMessage::getMidiNoteName(midi, true, true, 4); }

/** A dense-vocab RT token in words (onset ids are relative to `base`). */
auto rtTokenText(const juce::String &field, int token, int base) -> juce::String {
    if (field == "onset")
        return "onset " + juce::String(token + base) + " cs";
    if (field == "dur")
        return juce::String(token - static_cast<int>(DenseVocab::DurOffset)) + " cs";
    if (field == "pitch")
        return noteName((token - static_cast<int>(DenseVocab::NoteOffset)) % DenseConfig::MaxPitch);
    if (field == "vel")
        return "vel " + juce::String(token - static_cast<int>(DenseVocab::VelocityOffset));
    return {};
}

auto timeText(double ms) -> juce::String {
    const int total = static_cast<int>(ms / 1000.0);
    return juce::String::formatted("%d:%02d.%01d", total / 60, total % 60, static_cast<int>(std::fmod(ms, 1000.0) / 100.0));
}

auto pad(const juce::String &s, int width) -> juce::String { return s.paddedRight(' ', width); }

} // namespace

ReplayPage::ReplayPage(AppSession &liveSession) : live(liveSession) {
    addAndMakeVisible(toolbar);
    for (auto *c: std::initializer_list<juce::Component *>{&launchCombo, &sessionCombo, &refreshButton, &loadButton,
                                                            &closeButton, &filterEditor, &statusLabel})
        toolbar.addAndMakeVisible(c);
    launchCombo.setTextWhenNothingSelected("Choose a recorded launch");
    sessionCombo.setTextWhenNothingSelected("Session (Start -> Stop)");
    filterEditor.setTextToShowWhenEmpty("filter: op or text (e.g. RT_DRAW, k12)", kInk.withAlpha(0.5f));
    statusLabel.setColour(juce::Label::textColourId, kInk);
    refreshButton.onClick = [this] { refreshLaunches(); };
    loadButton.onClick = [this] { loadSelectedLaunch(); };
    closeButton.onClick = [this] { closeReplay(); };
    sessionCombo.onChange = [this] { selectSession(sessionCombo.getSelectedItemIndex()); };
    filterEditor.onTextChange = [this] { applyFilter(); };

    recordList.setRowHeight(18);
    recordList.setColour(juce::ListBox::backgroundColourId, UiConstants::pianoRollBackground);
    addAndMakeVisible(recordList);

    for (auto *view: {&inputsView, &drawView}) {
        view->setMultiLine(true);
        view->setReadOnly(true);
        view->setScrollbarsShown(true);
        view->setFont(mono());
        view->setColour(juce::TextEditor::backgroundColourId, UiConstants::pianoRollBackground);
        view->setColour(juce::TextEditor::textColourId, kInk);
    }
    rightTabs.addTab("Workspace", UiConstants::appRootTabColour, &workspaceHolder, false);
    rightTabs.addTab("Model inputs", UiConstants::appRootTabColour, &inputsView, false);
    rightTabs.addTab("Draw", UiConstants::appRootTabColour, &drawView, false);
    addAndMakeVisible(rightTabs);

    refreshLaunches();
    closeButton.setEnabled(false);
    startTimerHz(10);
}

ReplayPage::~ReplayPage() {
    stopTimer();
    closeReplay();
}

auto ReplayPage::paint(juce::Graphics &g) -> void {
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

auto ReplayPage::resized() -> void {
    auto area = getLocalBounds();
    auto bar = area.removeFromTop(34);
    toolbar.setBounds(bar);
    auto t = toolbar.getLocalBounds().reduced(4);
    launchCombo.setBounds(t.removeFromLeft(260));
    t.removeFromLeft(4);
    refreshButton.setBounds(t.removeFromLeft(70));
    t.removeFromLeft(4);
    loadButton.setBounds(t.removeFromLeft(60));
    t.removeFromLeft(4);
    closeButton.setBounds(t.removeFromLeft(100));
    t.removeFromLeft(8);
    sessionCombo.setBounds(t.removeFromLeft(260));
    t.removeFromLeft(8);
    filterEditor.setBounds(t.removeFromLeft(240));
    t.removeFromLeft(8);
    statusLabel.setBounds(t);

    auto left = area.removeFromLeft(area.getWidth() * 2 / 5);
    recordList.setBounds(left.reduced(4));
    rightTabs.setBounds(area.reduced(4));
    if (workspace != nullptr)
        workspace->setBounds(workspaceHolder.getLocalBounds());
}

// ---- launches, loading

auto ReplayPage::refreshLaunches() -> void {
    launches = OpLogReader::listLaunches(logsDir());
    launchCombo.clear(juce::dontSendNotification);
    int id = 1;
    for (const auto &l: launches) {
        juce::String text = l.time.toString(true, true, false, true);
        text << (l.oplog.existsAsFile() ? "" : "  (no oplog)") << (l.sessionLog.existsAsFile() ? "" : "  (no session log)");
        launchCombo.addItem(text, id++);
    }
}

auto ReplayPage::loadSelectedLaunch() -> void {
    const int i = launchCombo.getSelectedItemIndex();
    if (i < 0 || i >= static_cast<int>(launches.size()))
        return;
    if (live.isMusicThreadRunning()) {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon, "Generation is running",
                                               "Stop live generation before loading a replay.");
        return;
    }
    closeReplay();
    const auto &launch = launches[static_cast<size_t>(i)];
    statusLabel.setText("Loading " + launch.stamp + " ...", juce::dontSendNotification);
    if (launch.oplog.existsAsFile()) {
        log = OpLogReader::Log::load(launch.oplog);
        if (! log->error.isEmpty()) {
            statusLabel.setText(log->error, juce::dontSendNotification);
            log.reset();
        }
    }
    if (launch.sessionLog.existsAsFile()) {
        auto lines = LogReplay::readLog(launch.sessionLog);
        if (! lines.empty()) {
            live.replayActive = true;
            driver = std::make_unique<ReplayDriver>(std::move(lines));
            juce::Component::SafePointer<ReplayPage> self(this);
            driver->onSession = [self, this](AppSession *s) {
                if (s == nullptr)
                    return; // the page dropped the workspace before letting the session go
                const int e = epoch.load();
                juce::MessageManager::callAsync([self, s, e] {
                    if (self != nullptr && self->epoch.load() == e && self->driver != nullptr)
                        self->attachWorkspace(s);
                });
            };
        }
    }
    sessionCombo.clear(juce::dontSendNotification);
    if (log != nullptr) {
        int id = 1;
        for (const auto &s: log->sessions())
            sessionCombo.addItem(juce::String(s.run) + "   " + timeText(s.startMs) + " - " + timeText(s.endMs)
                                     + (s.stopReason.empty() ? "" : "  (" + juce::String(s.stopReason) + ")"),
                                 id++);
        if (! log->sessions().empty())
            sessionCombo.setSelectedItemIndex(0, juce::sendNotificationSync);
        else
            selectSession(-1);
    }
    closeButton.setEnabled(log != nullptr || driver != nullptr);
    if (log == nullptr && driver == nullptr)
        statusLabel.setText("Nothing to replay in " + launch.stamp, juce::dontSendNotification);
}

auto ReplayPage::closeReplay() -> void {
    detachWorkspace();
    driver.reset(); // after the workspace: its session goes now
    log.reset();
    rows.clear();
    recordList.updateContent();
    sessionCombo.clear(juce::dontSendNotification);
    inputsView.clear();
    drawView.clear();
    currentSession = -1;
    live.replayActive = false;
    closeButton.setEnabled(false);
    statusLabel.setText({}, juce::dontSendNotification);
}

auto ReplayPage::detachWorkspace() -> void {
    ++epoch;
    workspace.reset();
}

auto ReplayPage::attachWorkspace(AppSession *replaySession) -> void {
    workspace.reset();
    workspace = std::make_unique<WorkspacePage>(*replaySession);
    workspace->setViewOnly(true);
    workspaceHolder.addAndMakeVisible(*workspace);
    workspace->setBounds(workspaceHolder.getLocalBounds());
}

// ---- the record list

auto ReplayPage::selectSession(int sessionIndex) -> void {
    currentSession = sessionIndex;
    applyFilter();
    if (log != nullptr && sessionIndex >= 0 && sessionIndex < static_cast<int>(log->sessions().size()) && ! rows.empty())
        recordList.selectRow(0);
}

auto ReplayPage::applyFilter() -> void {
    rows.clear();
    if (log != nullptr) {
        size_t from = 0, to = log->records().size();
        if (currentSession >= 0 && currentSession < static_cast<int>(log->sessions().size())) {
            from = log->sessions()[static_cast<size_t>(currentSession)].first;
            to = log->sessions()[static_cast<size_t>(currentSession)].last + 1;
        }
        const auto filter = filterEditor.getText().trim().toStdString();
        for (size_t i = from; i < to; ++i) {
            if (! filter.empty()) {
                const auto &r = log->records()[i];
                if (r.op.find(filter) == std::string::npos && log->line(i).find(filter) == std::string::npos)
                    continue;
            }
            rows.push_back(i);
        }
    }
    recordList.updateContent();
    recordList.repaint();
}

auto ReplayPage::getNumRows() -> int { return static_cast<int>(rows.size()); }

auto ReplayPage::paintListBoxItem(int row, juce::Graphics &g, int width, int height, bool selected) -> void {
    if (log == nullptr || row < 0 || row >= static_cast<int>(rows.size()))
        return;
    const auto index = rows[static_cast<size_t>(row)];
    const auto &r = log->records()[index];
    if (selected)
        g.fillAll(juce::Colour(0xff2d4a63));
    g.setFont(mono());
    g.setColour(kInk.withAlpha(0.6f));
    g.drawText(timeText(r.t), 4, 0, 70, height, juce::Justification::centredLeft, false);
    g.drawText(r.clock >= 0 ? juce::String(r.clock) : juce::String("-"), 76, 0, 60, height, juce::Justification::centredLeft, false);
    g.setColour(kInk);
    g.drawText(r.op, 138, 0, 110, height, juce::Justification::centredLeft, false);
    // the record's own fields (what follows "op":"...")
    auto rest = juce::String(log->line(index));
    rest = rest.fromFirstOccurrenceOf("\"op\":\"" + juce::String(r.op) + "\"", false, false).trimCharactersAtStart(",").dropLastCharacters(1);
    g.setColour(kInk.withAlpha(0.75f));
    g.drawText(rest, 250, 0, width - 254, height, juce::Justification::centredLeft, true);
}

auto ReplayPage::selectedRowsChanged(int lastRowSelected) -> void {
    if (log == nullptr || lastRowSelected < 0 || lastRowSelected >= static_cast<int>(rows.size()))
        return;
    showRecord(rows[static_cast<size_t>(lastRowSelected)]);
}

auto ReplayPage::showRecord(size_t index) -> void {
    const double ms = log->records()[index].t;
    if (driver != nullptr) {
        // Going back drops the replay's session: the workspace goes first.
        if (ms < driver->position() - 0.5)
            detachWorkspace();
        driver->seek(ms);
    }
    inputsView.setText(inputsText(index), false);
    drawView.setText(drawText(index), false);
    const auto &op = log->records()[index].op;
    if (op == "RT_DRAW" || op == "OT_DRAW" || op == "RT_WAIT" || op == "VOCSEP" || op == "TD_CTRL")
        rightTabs.setCurrentTabIndex(2, false);
}

auto ReplayPage::timerCallback() -> void {
    if (driver == nullptr)
        return;
    juce::String s = "replay at " + timeText(driver->position());
    if (driver->busy())
        s << "  (catching up with " << timeText(driver->target()) << ")";
    if (driver->atEnd())
        s << "  (end of the session log)";
    if (workspace == nullptr)
        s << "  - workspace: waiting for the replay";
    statusLabel.setText(s, juce::dontSendNotification);
}

// ---- the text views

auto ReplayPage::inputsText(size_t index) const -> juce::String {
    const auto st = OpLogReader::stateAt(*log, index);
    juce::String out;
    for (const auto &[model, c]: st.rt) {
        const auto tokens = c.tokens();
        out << "RT " << model << "   base " << c.base << " cs, " << static_cast<int>(tokens.size()) << " tokens"
            << (c.exact ? "" : "   (approximate: a rebase since the last checkpoint)") << "\n";
        if (! c.prefix.empty()) {
            out << "  prefix:";
            for (const int t: c.prefix)
                out << " " << t;
            out << "\n";
        }
        for (const auto &item: c.items) {
            out << "  " << pad(item.kind, 8) << pad(item.id, 7);
            for (const int t: item.tokens)
                out << " " << t;
            out << "\n";
        }
        out << "\n";
    }
    if (st.rtCheckpointState.has_value()) {
        const auto f = log->fields(*st.rtCheckpointState);
        out << "RT state at the last checkpoint: " << juce::JSON::toString(f["state"], true) << "\n\n";
    }
    if (st.otInput.has_value()) {
        const auto f = log->fields(*st.otInput);
        const auto *notes = f["notes"].getArray();
        const auto *ids = f["input_ids"].getArray();
        const auto *combos = f["combo_in"].getArray();
        const int tpn = static_cast<int>(f["tokens_per_note"]);
        const int cl = static_cast<int>(f["combo_len"]);
        out << "OT encoder input (" << (notes != nullptr ? notes->size() : 0) << " notes, onsets from "
            << f["base_onset"].toString() << " cs)\n";
        if (notes != nullptr && ids != nullptr && combos != nullptr)
            for (int i = 0; i < notes->size(); ++i) {
                out << "  " << pad((*notes)[i].toString(), 8) << "tokens";
                for (int k = 0; k < tpn && i * tpn + k < ids->size(); ++k)
                    out << " " << (*ids)[i * tpn + k].toString();
                out << "   combo_in";
                for (int k = 0; k < cl && i * cl + k < combos->size(); ++k)
                    out << " " << (*combos)[i * cl + k].toString();
                out << "\n";
            }
        if (! st.otDrawsSinceInput.empty()) {
            out << "  drawn since (teacher-forced into combo_in as they come):\n";
            for (const auto d: st.otDrawsSinceInput) {
                const auto df = log->fields(d);
                out << "    " << pad(df["for"].toString(), 8) << "combo " << juce::JSON::toString(df["combo"], true) << "\n";
            }
        }
        out << "\n";
    }
    if (st.vocsep.has_value()) {
        const auto f = log->fields(*st.vocsep);
        out << "Vocsep window (last assignment: " << f["for"].toString() << " -> voice " << f["voice"].toString()
            << ", " << f["reason"].toString() << ")\n";
        if (const auto *w = f["window"].getArray())
            for (const auto &n: *w)
                out << "  onset " << pad(n[0].toString(), 7) << "dur " << pad(n[1].toString(), 5) << pad(noteName(static_cast<int>(n[2])), 5)
                    << "voice " << pad(n[3].toString(), 5) << n[4].toString() << "\n";
        out << "\n";
    }
    out << "Queues (waiting at this record)\n";
    for (const auto &[queue, items]: st.queues) {
        out << "  " << pad(queue, 14) << static_cast<int>(items.size()) << ":";
        for (const auto &item: items)
            out << " " << item.item;
        out << "\n";
    }
    return out;
}

auto ReplayPage::drawText(size_t index) const -> juce::String {
    const auto &r = log->records()[index];
    const auto f = log->fields(index);
    juce::String out;
    const auto table = [&](const juce::var &draw, bool rt, int base) {
        const auto field = draw["field"].toString();
        out << "  " << pad(field, 10) << "chosen " << draw["chosen"].toString();
        if (rt)
            out << " (" << rtTokenText(field, static_cast<int>(draw["chosen"]), base) << ")";
        out << "   p " << juce::String(static_cast<double>(draw["p"]), 4) << "   p_model "
            << juce::String(static_cast<double>(draw["p_model"]), 4) << (static_cast<bool>(draw["forced"]) ? "   (forced)" : "") << "\n";
        out << "      rank  token      " << (rt ? pad("meaning", 18) : juce::String()) << "p        p_model\n";
        if (const auto *top = draw["top"].getArray())
            for (int i = 0; i < top->size(); ++i) {
                const auto &e = (*top)[i];
                out << "      " << pad(juce::String(i + 1), 6) << pad(e[0].toString(), 11);
                if (rt)
                    out << pad(rtTokenText(field, static_cast<int>(e[0]), base), 18);
                out << pad(juce::String(static_cast<double>(e[1]), 4), 9) << juce::String(static_cast<double>(e[2]), 4) << "\n";
            }
        out << "\n";
    };
    if (r.op == "RT_DRAW") {
        out << "RT draw " << f["r"].toString() << " by " << f["model"].toString() << "   now " << f["now"].toString()
            << "   floor " << f["floor"].toString() << "   base " << f["base"].toString() << "   context "
            << f["context_len"].toString() << " tokens\n";
        out << "p = as sampled (masks, biases, temperature, top-p); p_model = the model's own softmax.\n"
            << "The top 10 are ranked by p.\n\n";
        if (const auto *fields = f["fields"].getArray())
            for (const auto &d: *fields)
                table(d, true, static_cast<int>(f["base"]));
    } else if (r.op == "OT_DRAW") {
        out << "OT draw for " << f["for"].toString() << "   pitch " << noteName(static_cast<int>(f["pitch"])) << "   onset "
            << f["onset"].toString() << "   instruments " << juce::JSON::toString(f["instruments"], true) << "\n"
            << "combo " << juce::JSON::toString(f["combo"], true) << "\n\n";
        if (const auto *steps = f["steps"].getArray())
            for (const auto &d: *steps)
                table(d, false, 0);
    } else if (r.op == "RT_WAIT") {
        out << "WAIT decision by " << f["model"].toString() << ": " << f["decision"].toString() << " at "
            << f["at"].toString() << "\n";
        if (f.hasProperty("p_wait"))
            out << "P(wait) " << juce::String(static_cast<double>(f["p_wait"]), 4) << "  (plays when P(play) >= 0.30)\n";
    } else if (r.op == "VOCSEP") {
        out << "Vocsep: " << f["for"].toString() << " -> voice " << f["voice"].toString() << " (" << f["reason"].toString()
            << "), parent " << f["parent"].toString() << ", score " << f["parent_score"].toString() << "\n"
            << "candidate parents (window index, score): " << juce::JSON::toString(f["candidates"], true) << "\n";
    } else if (r.op == "TD_CTRL") {
        out << "Dynamic LogitTD, model " << f["model"].toString() << ": target TD " << f["td"].toString() << ", actual "
            << f["measured"].toString() << ", error " << f["error"].toString() << ", gamma " << f["gamma"].toString()
            << ", K " << f["k"].toString() << ", S " << f["s_hat"].toString()
            << (static_cast<bool>(f["frozen"]) ? ", frozen" : "") << " " << f["why"].toString() << "\n";
    } else {
        out << "Select an RT_DRAW, OT_DRAW, RT_WAIT, VOCSEP or TD_CTRL record to see its details.\n\n";
    }
    out << "\nRecord:\n" << juce::JSON::toString(f, false);
    return out;
}
