// OrchestraControls.cpp developed by Bob Tianqi Wei
#include "VirtualOrch/widgets/OrchestraControls.h"

#include "VirtualOrch/AppSession.h"
#include "VirtualOrch/OSCController.h"
#include "VirtualOrch/Sympathetic.h"
#include "VirtualOrch/ui/UiConstants.h"

namespace {

using juce::File;
using juce::String;

auto sourceDirectory() -> File {
    return File(VIRTUAL_ORCH_APP_DIR).getSiblingFile("sympathetic");
}

auto resourceDirectory() -> File {
    const auto binaryDirectory = File::getSpecialLocation(File::currentExecutableFile).getParentDirectory();
#if JUCE_MAC
    const auto bundled = binaryDirectory.getSiblingFile("Resources").getChildFile("sympathetic");
#else
    const auto bundled = binaryDirectory.getChildFile("sympathetic");
#endif
    return bundled.isDirectory() ? bundled : sourceDirectory();
}

auto findProgram(const String &name, const String &overrideName = {}) -> String {
    if (overrideName.isNotEmpty()) {
        const auto configured = juce::SystemStats::getEnvironmentVariable(overrideName, {});
        if (configured.isNotEmpty())
            return configured;
    }

    auto directories = juce::StringArray::fromTokens(
        juce::SystemStats::getEnvironmentVariable("PATH", ""), ":", {});
    directories.addArray({"/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/opt/anaconda3/bin"});
    for (const auto &directory: directories) {
        if (! File::isAbsolutePath(directory))
            continue;
        const auto candidate = File(directory).getChildFile(name);
        if (candidate.existsAsFile())
            return candidate.getFullPathName();
    }
    return {};
}

auto arucoPython() -> String {
    const auto configured = juce::SystemStats::getEnvironmentVariable("VIRTUAL_ORCH_ARUCO_PYTHON", {});
    if (configured.isNotEmpty())
        return configured;

    for (const auto &dir: {sourceDirectory(), resourceDirectory()}) {
        const auto virtualEnv = dir.getChildFile("aruco/.venv/bin/python");
        if (virtualEnv.existsAsFile())
            return virtualEnv.getFullPathName();
    }
    return findProgram("python3");
}

class SessionDirectory {
public:
    explicit SessionDirectory(const String &suffix)
        : directory(File::createTempFile(suffix)) {}

    ~SessionDirectory() { directory.deleteRecursively(); }

    const File directory;
};

void prepareStatus(juce::Label &label) {
    label.setMinimumHorizontalScale(1.0f);
    label.setJustificationType(juce::Justification::centredLeft);
}

} // namespace

class OrchestraControls::ConductingPage final : public juce::Component, private juce::Timer {
public:
    explicit ConductingPage(AppSession &sessionIn) : session(sessionIn) {
        prepareStatus(status);
        addAndMakeVisible(status);
        addChildComponent(openBrowser);
        openBrowser.onClick = [this] { openInBrowser(); };
    }

    ~ConductingPage() override { stop(); }

    void setSelected(bool selected) {
        const bool entered = selected && ! wasSelected;
        wasSelected = selected;
        if (entered && ! isTimerRunning())
            start();
    }

    void resized() override {
        auto area = getLocalBounds();
#if JUCE_WEB_BROWSER
        if (browser != nullptr) {
            browser->setBounds(area);
            return;
        }
#endif
        area = area.reduced(8);
        status.setBounds(area.removeFromTop(28));
        if (openBrowser.isVisible())
            openBrowser.setBounds(area.removeFromTop(26).removeFromLeft(126));
    }

private:
#if JUCE_WEB_BROWSER
    class Browser final : public juce::WebBrowserComponent {
    public:
        Browser(String allowed, String open, std::function<void()> onOpen)
            : juce::WebBrowserComponent(Options{}.withKeepPageLoadedWhenBrowserIsHidden()),
              allowedUrl(std::move(allowed)),
              openUrl(std::move(open)),
              openExternally(std::move(onOpen)) {}

        bool pageAboutToLoad(const String &url) override {
            if (url == openUrl) {
                juce::MessageManager::callAsync(openExternally);
                return false;
            }
            return url == allowedUrl || url == "about:blank";
        }

    private:
        const String allowedUrl, openUrl;
        const std::function<void()> openExternally;
    };
#endif

    void openInBrowser() {
        if (page.isEmpty())
            return;
#if JUCE_WEB_BROWSER
        browser.reset();
#endif
        status.setVisible(true);
        status.setText("Conducting is open in your browser.", juce::dontSendNotification);
        openBrowser.setVisible(true);
        resized();
        juce::URL(page).launchInDefaultBrowser();
    }

    void start() {
        status.setText("Starting Conducting…", juce::dontSendNotification);
        status.setVisible(true);
        openBrowser.setVisible(false);
        scratch.directory.createDirectory();
        ready.deleteFile();

        const auto node = findProgram("node", "VIRTUAL_ORCH_NODE");
        if (node.isEmpty()) {
            status.setText("Node.js is required for Conducting.", juce::dontSendNotification);
            return;
        }

        socket = std::make_unique<juce::DatagramSocket>();
        if (! socket->bindToPort(0, "127.0.0.1")) {
            status.setText("Could not open the local gesture receiver.", juce::dontSendNotification);
            return;
        }
        controller = std::make_unique<OSCController>(*socket);
        controller->onSympatheticGesture = [this] { receivedGesture = true; };
        controller->onSympatheticPedal = [this](bool down) {
            stoppedByGesture = down;
            session.midiInputProcess.setPedal(down, down ? "Sympathetic stop" : "Sympathetic play");
        };

        token = juce::Uuid().toString();
        const auto script = resourceDirectory().getChildFile("sym_server.js");
        if (! script.existsAsFile()
            || ! relay.start({node, script.getFullPathName(), "--embedded", token,
                              String(socket->getBoundPort()), ready.getFullPathName()}, 0)) {
            stop();
            status.setText("Could not start the Conducting relay.", juce::dontSendNotification);
            return;
        }
        startTime = juce::Time::getMillisecondCounter();
        startTimer(100);
    }

    void stop() {
        stopTimer();
#if JUCE_WEB_BROWSER
        browser.reset();
#endif
        if (relay.isRunning()) {
            relay.kill();
            relay.waitForProcessToFinish(500);
        }
        controller.reset();
        socket.reset();
        ready.deleteFile();
        ready.withFileExtension(".port.tmp").deleteFile();
        page.clear();
        openBrowser.setVisible(false);
        status.setVisible(true);
        resized();
        if (stoppedByGesture)
            session.midiInputProcess.setPedal(false, "Conducting closed");
        if (receivedGesture) {
            auto &state = Sympathetic::state();
            state.enabled.store(false);
            state.reset();
        }
        receivedGesture = false;
        stoppedByGesture = false;
    }

    void timerCallback() override {
        if (! relay.isRunning() || (page.isEmpty() && juce::Time::getMillisecondCounter() - startTime > 8000)) {
            stop();
            status.setText("Conducting relay could not start. Switch tabs to retry.",
                           juce::dontSendNotification);
            return;
        }
        if (page.isNotEmpty())
            return;

        const int port = ready.loadFileAsString().getIntValue();
        if (port <= 0)
            return;

        page = "http://127.0.0.1:" + String(port) + "/" + token + "/conduct.html";
#if JUCE_WEB_BROWSER
        const auto embeddedPage = page + "?embedded=1";
        const auto openUrl = page.upToLastOccurrenceOf("/", true, false) + "open-in-browser";
        const juce::Component::SafePointer<ConductingPage> safe(this);
        browser = std::make_unique<Browser>(embeddedPage, openUrl, [safe] {
            if (safe != nullptr)
                safe->openInBrowser();
        });
        status.setVisible(false);
        addAndMakeVisible(*browser);
        resized();
        browser->goToURL(embeddedPage);
#else
        status.setText("Use a browser for Conducting on this build.", juce::dontSendNotification);
        openBrowser.setVisible(true);
        resized();
#endif
    }

    AppSession &session;
    SessionDirectory scratch{".conducting"};
    const File ready = scratch.directory.getChildFile("ready.port");
    juce::ChildProcess relay;
    std::unique_ptr<juce::DatagramSocket> socket;
    std::unique_ptr<OSCController> controller;
#if JUCE_WEB_BROWSER
    std::unique_ptr<Browser> browser;
#endif
    juce::Label status;
    juce::TextButton openBrowser{"Open in Browser"};
    String token, page;
    uint32_t startTime = 0;
    bool wasSelected = false;
    bool receivedGesture = false;
    bool stoppedByGesture = false;
};

class OrchestraControls::TangiblePage final : public juce::Component, private juce::Timer {
public:
    TangiblePage() {
        camera.setTextWhenNothingSelected("No camera selected");
        camera.setTitle("Camera");
        cameraButton.setEnabled(false);
        paper.addItem("US Letter", 1);
        paper.addItem("A4", 2);
        paper.setSelectedId(1);
        paper.setTitle("Paper");
        markerSize.setText("45");
        markerSize.setInputRestrictions(5, "0123456789.");
        markerSize.setTitle("Marker size in millimetres");
        markerSize.setTooltip("Width of the black square, in millimetres.");
        prepareStatus(status);
        prepareStatus(parts);
        status.setText("Start the camera to enable marker control.", juce::dontSendNotification);
        parts.setJustificationType(juce::Justification::topLeft);
        parts.setText("No markers detected", juce::dontSendNotification);
        markerList.setViewedComponent(&parts, false);
        markerList.setScrollBarsShown(true, false);

        for (auto *component: std::initializer_list<juce::Component *>{
                 &preview, &cameraLabel, &camera, &refreshCameras, &cameraButton, &paper, &markerSize, &sizeLabel,
                 &printButton, &status, &markerList})
            addAndMakeVisible(component);
        cameraButton.onClick = [this] {
            if (tracking != nullptr)
                stopTracking();
            else
                startTracking();
        };
        refreshCameras.onClick = [this] { scanCameras(); };
        printButton.onClick = [this] { choosePdf(); };
    }

    ~TangiblePage() override {
        stopTracking();
        finish(printing);
        finish(scanning);
        stopTimer();
    }

    void setSelected(bool selected) {
        if (selected && ! wasSelected && tracking == nullptr)
            scanCameras();
        wasSelected = selected;
    }

    void stopTracking() {
        if (tracking == nullptr)
            return;

        Sympathetic::state().tangibleMask.store(Sympathetic::kTangibleInactive);
        finish(tracking);
        frameFile.deleteFile();
        frameFile.withFileExtension(".tmp").deleteFile();
        preview.setImage({});
        cameraButton.setButtonText("Start Camera");
        camera.setEnabled(true);
        refreshCameras.setEnabled(true);
        showMarkers({});
        status.setText("Camera stopped. Manual instrument controls restored.", juce::dontSendNotification);
    }

    void resized() override {
        auto area = getLocalBounds().reduced(8);
        constexpr int gap = 6;
        constexpr int rowHeight = 28;
        constexpr int fixedWidths = 52 + 66 + 104 + 82 + 72 + 116 + gap * 6;
        const auto cameraWidth = juce::jlimit(120, 180, area.getWidth() - fixedWidths);
        auto row = area.removeFromTop(rowHeight);
        const auto take = [&](int width) {
            if (row.getWidth() < width && row.getX() > area.getX()) {
                area.removeFromTop(gap);
                row = area.removeFromTop(rowHeight);
            }
            auto bounds = row.removeFromLeft(width);
            row.removeFromLeft(gap);
            return bounds;
        };

        auto cameraField = take(52 + cameraWidth);
        cameraLabel.setBounds(cameraField.removeFromLeft(52));
        camera.setBounds(cameraField);
        refreshCameras.setBounds(take(66));
        cameraButton.setBounds(take(104));
        paper.setBounds(take(82));
        auto sizeField = take(72);
        markerSize.setBounds(sizeField.removeFromLeft(44));
        sizeLabel.setBounds(sizeField);
        printButton.setBounds(take(116));

        area.removeFromTop(gap);
        auto details = area.removeFromRight(juce::jmin(228, area.getWidth() * 2 / 5));
        area.removeFromRight(gap);
        status.setBounds(details.removeFromTop(46));
        markerList.setBounds(details);
        resizeMarkerList();
        preview.setBounds(area);
    }

private:
    static void finish(std::unique_ptr<juce::ChildProcess> &process) {
        if (process != nullptr && process->isRunning()) {
            process->kill();
            process->waitForProcessToFinish(500);
        }
        process.reset();
    }

    bool prepareWorker() {
        const auto directoryResult = scratch.directory.createDirectory();
        if (directoryResult.failed()) {
            status.setText(directoryResult.getErrorMessage(), juce::dontSendNotification);
            return false;
        }

        juce::Array<juce::var> entries;
        int id = 1;
        for (const auto &part: Sympathetic::kMarkerParts) {
            auto entry = std::make_unique<juce::DynamicObject>();
            entry->setProperty("id", id++);
            entry->setProperty("part", String(part.name.data()));
            entries.add(juce::var(entry.release()));
        }
        if (! manifest.replaceWithText(juce::JSON::toString(juce::var(entries)))) {
            status.setText("Could not write the marker assignments.", juce::dontSendNotification);
            return false;
        }
        return true;
    }

    auto runWorker(const String &mode, const File &result, const juce::StringArray &options)
        -> std::unique_ptr<juce::ChildProcess> {
        const auto python = arucoPython();
        if (python.isEmpty() || ! prepareWorker()) {
            if (python.isEmpty())
                status.setText("Python 3 with OpenCV and ReportLab is required.", juce::dontSendNotification);
            return nullptr;
        }

        result.deleteFile();
        const auto script = resourceDirectory().getChildFile("aruco/worker.py");
        juce::StringArray args{python, "-B", script.getFullPathName(), mode,
                               manifest.getFullPathName(), result.getFullPathName()};
        args.addArray(options);
        auto worker = std::make_unique<juce::ChildProcess>();
        if (! script.existsAsFile() || ! worker->start(args, 0)) {
            status.setText("Could not start the ArUco worker.", juce::dontSendNotification);
            return nullptr;
        }
        startTimer(70);
        return worker;
    }

    void scanCameras() {
        if (tracking != nullptr || scanning != nullptr)
            return;

        scanning = runWorker("devices", deviceResult, {});
        if (scanning != nullptr) {
            refreshCameras.setEnabled(false);
            cameraButton.setEnabled(false);
            status.setText("Looking for cameras…", juce::dontSendNotification);
        }
    }

    void updateCameras() {
        const auto result = juce::JSON::parse(deviceResult);
        const auto previous = cameraIds[camera.getSelectedItemIndex()];
        camera.clear(juce::dontSendNotification);
        cameraIds.clear();
        const auto devices = result.getProperty("devices", {});
        if (const auto *list = devices.getArray()) {
            for (const auto &device: *list) {
                cameraIds.add(device.getProperty("id", {}).toString());
                camera.addItem(device.getProperty("name", {}).toString(), cameraIds.size());
            }
        }

        if (cameraIds.isEmpty()) {
            status.setText(result.getProperty("error", "No cameras found. Connect one and refresh.").toString(),
                           juce::dontSendNotification);
        } else {
            camera.setSelectedItemIndex(juce::jmax(0, cameraIds.indexOf(previous)),
                                        juce::dontSendNotification);
            status.setText("Start the camera to enable marker control.", juce::dontSendNotification);
        }

        scanning.reset();
        refreshCameras.setEnabled(true);
        cameraButton.setEnabled(! cameraIds.isEmpty());
    }

    void startTracking() {
        const auto selected = camera.getSelectedItemIndex();
        if (! juce::isPositiveAndBelow(selected, cameraIds.size()))
            return;

        tracking = runWorker("track", frameFile, {"--camera", cameraIds[selected]});
        if (tracking == nullptr)
            return;

        lastFrame = 0;
        cameraWaiting = true;
        lastFrameTime = juce::Time::getMillisecondCounter();
        cameraButton.setButtonText("Stop Camera");
        camera.setEnabled(false);
        refreshCameras.setEnabled(false);
        status.setText("Starting camera…", juce::dontSendNotification);
    }

    void choosePdf() {
        if (!(markerSize.getText().getDoubleValue() > 0.0)) {
            status.setText("Enter a marker size greater than zero.", juce::dontSendNotification);
            return;
        }
        chooser = std::make_unique<juce::FileChooser>(
            "Save Printable Markers",
            File::getSpecialLocation(File::userDocumentsDirectory).getChildFile("Orchestra Markers.pdf"),
            "*.pdf");
        chooser->launchAsync(juce::FileBrowserComponent::saveMode
                                 | juce::FileBrowserComponent::canSelectFiles
                                 | juce::FileBrowserComponent::warnAboutOverwriting,
            [safe = juce::Component::SafePointer<TangiblePage>(this)](const juce::FileChooser &dialog) {
                if (safe == nullptr || dialog.getResult() == File())
                    return;
                safe->pdfFile = dialog.getResult().withFileExtension(".pdf");
                safe->printing = safe->runWorker("pdf", safe->printResult,
                    {"--pdf", safe->pdfFile.getFullPathName(), "--paper", safe->paper.getText(),
                     "--size", safe->markerSize.getText()});
                if (safe->printing != nullptr) {
                    safe->printButton.setEnabled(false);
                    safe->status.setText("Creating printable markers…", juce::dontSendNotification);
                }
            });
    }

    void resizeMarkerList() {
        const int rowHeight = juce::roundToInt(parts.getFont().getHeight()) + 4;
        const int contentHeight = markerLineCount * rowHeight + parts.getBorderSize().getTopAndBottom();
        const int scrollbar = contentHeight > markerList.getHeight() ? markerList.getScrollBarThickness() : 0;
        parts.setSize(juce::jmax(0, markerList.getWidth() - scrollbar), contentHeight);
    }

    void showMarkers(const juce::StringArray &lines) {
        markerLineCount = juce::jmax(1, lines.size());
        parts.setText(lines.isEmpty() ? "No markers detected" : lines.joinIntoString("\n"),
                      juce::dontSendNotification);
        resizeMarkerList();
    }

    void readTracking() {
        const auto frame = juce::JSON::parse(frameFile);
        const auto now = juce::Time::getMillisecondCounter();
        const auto error = frame.getProperty("error", {}).toString();
        if (error.isNotEmpty() || ! tracking->isRunning()) {
            stopTracking();
            status.setText(error.isNotEmpty() ? error : "ArUco tracking ended.",
                           juce::dontSendNotification);
            return;
        }

        const int frameNumber = frame.getProperty("frame", 0);
        if (frameNumber == lastFrame) {
            if (lastFrame != 0 && now - lastFrameTime > 750 && ! cameraWaiting) {
                cameraWaiting = true;
                Sympathetic::state().tangibleMask.store(0);
                showMarkers({});
                status.setText("Waiting for the camera feed.", juce::dontSendNotification);
            } else if (lastFrame == 0 && now - lastFrameTime > 45000) {
                stopTracking();
                status.setText("The camera did not send a frame. Check access or choose another device.",
                               juce::dontSendNotification);
            }
            return;
        }

        if (cameraWaiting) {
            cameraWaiting = false;
            status.setText("Show a marker to enable its part.", juce::dontSendNotification);
        }
        lastFrame = frameNumber;
        lastFrameTime = now;
        uint32_t mask = 0;
        const auto ids = frame.getProperty("ids", {});
        if (const auto *array = ids.getArray()) {
            for (const auto &value: *array)
                mask |= Sympathetic::markerMask(static_cast<int>(value));
        }
        Sympathetic::state().tangibleMask.store(mask);
        juce::StringArray markerLines;
        const auto handStatus = frame.getProperty("hand_status", {}).toString();
        if (handStatus.isNotEmpty())
            markerLines.add(handStatus);
        const auto markers = frame.getProperty("markers", {});
        if (const auto *array = markers.getArray()) {
            for (const auto &marker: *array) {
                const int id = marker.getProperty("id", 0);
                if (Sympathetic::markerMask(id) == 0)
                    continue;
                const auto &part = Sympathetic::kMarkerParts[static_cast<size_t>(id - 1)];
                const int angle = marker.getProperty("angle", 0);
                const auto hand = marker.getProperty("hand", {}).toString();
                markerLines.add(String(id) + "  " + String(part.name.data()));
                markerLines.add(String(angle) + String::charToString(0x00b0)
                                + (hand.isEmpty() ? String() : "  |  " + hand));
            }
        }
        showMarkers(markerLines);

        juce::MemoryOutputStream decoded;
        if (juce::Base64::convertFromBase64(decoded, frame.getProperty("preview", {}).toString()))
            preview.setImage(juce::ImageFileFormat::loadFrom(decoded.getData(), decoded.getDataSize()),
                             juce::RectanglePlacement::centred);
    }

    void timerCallback() override {
        if (tracking != nullptr)
            readTracking();
        if (printing != nullptr && ! printing->isRunning()) {
            const auto result = juce::JSON::parse(printResult);
            printing.reset();
            printButton.setEnabled(true);
            if (static_cast<bool>(result.getProperty("ok", false))) {
                status.setText("Printable markers saved.", juce::dontSendNotification);
                pdfFile.startAsProcess();
            } else {
                status.setText(result.getProperty("error", "Could not create the marker PDF.").toString(),
                               juce::dontSendNotification);
            }
        }
        if (scanning != nullptr && ! scanning->isRunning())
            updateCameras();
        if (tracking == nullptr && printing == nullptr && scanning == nullptr)
            stopTimer();
    }

    SessionDirectory scratch{".tangible"};
    const File manifest = scratch.directory.getChildFile("markers.json");
    const File frameFile = scratch.directory.getChildFile("frame.json");
    const File printResult = scratch.directory.getChildFile("print.json");
    const File deviceResult = scratch.directory.getChildFile("devices.json");
    File pdfFile;
    std::unique_ptr<juce::ChildProcess> tracking, printing, scanning;
    std::unique_ptr<juce::FileChooser> chooser;
    juce::ImageComponent preview;
    juce::Label cameraLabel{{}, "Camera"}, sizeLabel{{}, "mm"}, status, parts;
    juce::Viewport markerList;
    int markerLineCount = 1;
    juce::TextEditor markerSize;
    juce::ComboBox camera, paper;
    juce::TextButton cameraButton{"Start Camera"}, refreshCameras{"Refresh"}, printButton{"Save Marker PDF"};
    juce::StringArray cameraIds;
    bool wasSelected = false;
    bool cameraWaiting = false;
    int lastFrame = 0;
    uint32_t lastFrameTime = 0;
};

OrchestraControls::OrchestraControls(AppSession &session, juce::Component &seatingPlan)
    : juce::TabbedComponent(juce::TabbedButtonBar::TabsAtTop),
      conducting(std::make_unique<ConductingPage>(session)),
      tangible(std::make_unique<TangiblePage>()) {
    setOutline(0);
    setIndent(0);
    setTabBarDepth(26);
    const auto colour = UiConstants::workspaceWidgetTitleBarBackground;
    addTab("Orchestra", colour, &seatingPlan, false);
    addTab("Conducting", colour, conducting.get(), false);
    addTab("Tangible", colour, tangible.get(), false);
}

OrchestraControls::~OrchestraControls() {
    clearTabs();
}

void OrchestraControls::currentTabChanged(int, const String &) {
    updateVisibility();
}

void OrchestraControls::updateVisibility() {
    const bool shown = isShowing();
    conducting->setSelected(shown && getCurrentTabIndex() == 1);
    tangible->setSelected(shown && getCurrentTabIndex() == 2);
}
