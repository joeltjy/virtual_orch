#include "VirtualOrch/OpLog.h"

#include <array>
#include <cmath>
#include <cstdio>

namespace OpLog {

namespace {

const char *srcName(Src s) {
    switch (s) {
        case Src::Midi: return "midi";
        case Src::Input: return "input";
        case Src::Rt: return "rt";
        case Src::Ot: return "ot";
        case Src::Play: return "play";
        case Src::Ui: return "ui";
        case Src::State: return "state";
    }
    return "ui";
}

auto number(double v) -> std::string {
    if (! std::isfinite(v))
        return "null";
    char buf[32];
    // up to 4 decimals, trailing zeros trimmed (probabilities, ms)
    std::snprintf(buf, sizeof(buf), "%.4f", v);
    std::string s(buf);
    if (s.find('.') != std::string::npos) {
        while (! s.empty() && s.back() == '0')
            s.pop_back();
        if (! s.empty() && s.back() == '.')
            s.pop_back();
    }
    return s;
}

struct State {
    std::mutex mutex; // guards buffer, capture, file
    std::vector<Record> buffer;
    std::function<void(const std::string &)> capture;
    std::function<int32_t()> clock;
    std::function<double()> time;
    std::atomic<bool> fileOpen{false};
    std::atomic<bool> capturing{false};
    std::atomic<int64_t> seq{0};
    juce::File file;
    std::unique_ptr<juce::OutputStream> out;
    double startMs = 0.0;
    std::atomic<int64_t> pads{0}, outputs{0}, runs{0};
    std::array<std::atomic<double>, 7> lastCheckpoint{};
};

auto state() -> State & {
    static State s;
    return s;
}

class Writer : public juce::Thread {
public:
    Writer() : juce::Thread("OpLog writer") {}
    auto run() -> void override {
        while (! threadShouldExit()) {
            wait(100);
            flush();
        }
        flush();
    }
    static auto flush() -> void {
        auto &s = state();
        std::vector<Record> batch;
        {
            const std::lock_guard<std::mutex> lock(s.mutex);
            batch.swap(s.buffer);
        }
        if (batch.empty() || s.out == nullptr)
            return;
        std::string text;
        text.reserve(batch.size() * 160);
        for (auto &r: batch) {
            text += format(r, s.seq.fetch_add(1));
            text += '\n';
        }
        s.out->write(text.data(), text.size());
        s.out->flush();
    }
};

auto writer() -> std::unique_ptr<Writer> & {
    static std::unique_ptr<Writer> w;
    return w;
}

} // namespace

auto quote(const std::string &in) -> std::string {
    std::string s;
    s.reserve(in.size() + 2);
    s += '"';
    for (const unsigned char c: in) {
        switch (c) {
            case '"': s += "\\\""; break;
            case '\\': s += "\\\\"; break;
            case '\n': s += "\\n"; break;
            case '\r': s += "\\r"; break;
            case '\t': s += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    s += buf;
                } else {
                    s += static_cast<char>(c);
                }
        }
    }
    s += '"';
    return s;
}

auto valueJson(const Value &value) -> std::string {
    struct V {
        std::string operator()(std::monostate) const { return "null"; }
        std::string operator()(bool b) const { return b ? "true" : "false"; }
        std::string operator()(int64_t i) const { return std::to_string(i); }
        std::string operator()(double d) const { return number(d); }
        std::string operator()(const std::string &s) const { return quote(s); }
        std::string operator()(const Ints &v) const {
            std::string s = "[";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i)
                    s += ',';
                s += std::to_string(v[i]);
            }
            return s + "]";
        }
        std::string operator()(const Floats &v) const {
            std::string s = "[";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i)
                    s += ',';
                s += number(v[i]);
            }
            return s + "]";
        }
        std::string operator()(const Strings &v) const {
            std::string s = "[";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i)
                    s += ',';
                s += quote(v[i]);
            }
            return s + "]";
        }
        std::string operator()(const Raw &r) const { return r.json.empty() ? "null" : r.json; }
    };
    return std::visit(V{}, value);
}

auto Obj::add(const char *key, const Value &value) -> Obj & {
    if (! body.empty())
        body += ',';
    body += quote(key);
    body += ':';
    body += valueJson(value);
    return *this;
}

auto format(const Record &r, int64_t seq) -> std::string {
    std::string s;
    s.reserve(160);
    s += "{\"seq\":";
    s += std::to_string(seq);
    s += ",\"t\":";
    s += number(r.t);
    s += ",\"clock\":";
    s += std::to_string(r.clock);
    s += ",\"src\":\"";
    s += srcName(r.src);
    s += "\",\"op\":\"";
    s += r.op;
    s += '"';
    for (const auto &f: r.fields) {
        if (std::holds_alternative<std::monostate>(f.value))
            continue; // absent
        s += ',';
        s += quote(f.key);
        s += ':';
        s += valueJson(f.value);
    }
    s += '}';
    return s;
}

auto on() -> bool {
    auto &s = state();
    return s.fileOpen.load(std::memory_order_relaxed) || s.capturing.load(std::memory_order_relaxed);
}

auto emit(Src src, const char *op, std::vector<Field> fields) -> void {
    if (! on())
        return;
    auto &s = state();
    Record r;
    r.t = s.time ? s.time() : juce::Time::getMillisecondCounterHiRes() - s.startMs;
    r.clock = s.clock ? s.clock() : -1;
    r.src = src;
    r.op = op;
    r.fields = std::move(fields);
    const std::lock_guard<std::mutex> lock(s.mutex);
    if (s.capture) {
        // capture: format now, in call order (the replay is single-threaded)
        s.capture(format(r, s.seq.fetch_add(1)));
        return;
    }
    if (s.fileOpen.load(std::memory_order_relaxed))
        s.buffer.push_back(std::move(r));
}

auto start(const juce::File &file) -> void {
    stop();
    auto &s = state();
    file.getParentDirectory().createDirectory();
    auto fos = std::make_unique<juce::FileOutputStream>(file);
    if (! fos->openedOk())
        return;
    fos->setPosition(0);
    fos->truncate();
    {
        const std::lock_guard<std::mutex> lock(s.mutex);
        s.file = file;
        s.out = std::make_unique<juce::GZIPCompressorOutputStream>(fos.release(), 6, true, juce::GZIPCompressorOutputStream::windowBitsGZIP);
        s.startMs = juce::Time::getMillisecondCounterHiRes();
        s.seq = 0;
        s.buffer.clear();
    }
    resetCounters();
    requestCheckpoints();
    s.fileOpen = true;
    writer() = std::make_unique<Writer>();
    writer()->startThread();
}

auto stop() -> void {
    auto &s = state();
    if (! s.fileOpen.load())
        return;
    emit(Src::Ui, "SESSION_END", {{"ops", s.seq.load()}});
    if (writer() != nullptr) {
        writer()->signalThreadShouldExit();
        writer()->notify();
        writer()->stopThread(3000);
        writer().reset();
    }
    Writer::flush();
    s.fileOpen = false;
    const std::lock_guard<std::mutex> lock(s.mutex);
    s.out.reset(); // finishes the gzip stream
}

auto currentFile() -> juce::File {
    return state().file;
}

auto setClock(std::function<int32_t()> clockCs) -> void {
    const std::lock_guard<std::mutex> lock(state().mutex);
    state().clock = std::move(clockCs);
}

auto setTime(std::function<double()> sessionMs) -> void {
    const std::lock_guard<std::mutex> lock(state().mutex);
    state().time = std::move(sessionMs);
}

auto setCapture(std::function<void(const std::string &line)> capture) -> void {
    auto &s = state();
    {
        const std::lock_guard<std::mutex> lock(s.mutex);
        s.capture = std::move(capture);
        s.capturing = s.capture != nullptr;
        if (s.capturing)
            s.seq = 0;
    }
    if (s.capturing) {
        resetCounters();
        requestCheckpoints();
    }
}

auto nextPadId() -> int64_t { return state().pads.fetch_add(1); }
auto nextOutputId() -> int64_t { return state().outputs.fetch_add(1); }
auto nextRunId() -> int64_t { return state().runs.fetch_add(1); }

auto resetCounters() -> void {
    auto &s = state();
    s.pads = 0;
    s.outputs = 0;
    s.runs = 0;
}

auto checkpointDue(Src part, double nowMs) -> bool {
    const auto last = state().lastCheckpoint[static_cast<size_t>(part)].load(std::memory_order_relaxed);
    return last < 0.0 || nowMs - last >= kCheckpointEveryMs;
}

auto markCheckpoint(Src part, double nowMs) -> void {
    state().lastCheckpoint[static_cast<size_t>(part)].store(nowMs, std::memory_order_relaxed);
}

auto requestCheckpoints() -> void {
    for (auto &c: state().lastCheckpoint)
        c.store(-1.0, std::memory_order_relaxed);
}

} // namespace OpLog
