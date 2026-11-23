#include "VirtualOrch/sim/OpLogReader.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace OpLogReader {

namespace {

/** "2026-11-21_15-47-27" (optionally "_2") -> time; invalid -> 0. */
auto timeOfStamp(const juce::String &stamp) -> juce::Time {
    const auto parts = juce::StringArray::fromTokens(stamp, "-_", "");
    if (parts.size() < 6)
        return {};
    return juce::Time(parts[0].getIntValue(), parts[1].getIntValue() - 1, parts[2].getIntValue(), parts[3].getIntValue(),
                      parts[4].getIntValue(), parts[5].getIntValue());
}

auto stampOf(const juce::File &f, const juce::String &prefix, const juce::String &suffix) -> juce::String {
    auto name = f.getFileName();
    if (! name.startsWith(prefix) || ! name.endsWith(suffix))
        return {};
    return name.substring(prefix.length(), name.length() - suffix.length());
}

/** The string value of `"key":"..."` in a JSON line (no escapes inside: ids and queue names). */
auto stringField(const std::string &line, const char *key) -> std::string {
    const std::string needle = std::string("\"") + key + "\":\"";
    const auto at = line.find(needle);
    if (at == std::string::npos)
        return {};
    const auto from = at + needle.size();
    const auto to = line.find('"', from);
    return to == std::string::npos ? std::string() : line.substr(from, to - from);
}

auto intsOf(const juce::var &v) -> std::vector<int> {
    std::vector<int> out;
    if (const auto *a = v.getArray())
        for (const auto &x: *a)
            out.push_back(static_cast<int>(x));
    return out;
}

/** Header of one line: {"seq":N,"t":X,"clock":C,"src":"S","op":"O" ... */
auto parseHeader(const char *p, const char *end, Record &r) -> bool {
    const auto expect = [&](const char *lit) {
        const size_t n = std::strlen(lit);
        if (static_cast<size_t>(end - p) < n || std::memcmp(p, lit, n) != 0)
            return false;
        p += n;
        return true;
    };
    char *after = nullptr;
    if (! expect("{\"seq\":"))
        return false;
    r.seq = std::strtoll(p, &after, 10);
    p = after;
    if (! expect(",\"t\":"))
        return false;
    r.t = std::strtod(p, &after);
    p = after;
    if (! expect(",\"clock\":"))
        return false;
    r.clock = static_cast<int32_t>(std::strtol(p, &after, 10));
    p = after;
    if (! expect(",\"src\":\""))
        return false;
    const char *q = static_cast<const char *>(std::memchr(p, '"', static_cast<size_t>(end - p)));
    if (q == nullptr)
        return false;
    r.src.assign(p, q);
    p = q + 1;
    if (! expect(",\"op\":\""))
        return false;
    q = static_cast<const char *>(std::memchr(p, '"', static_cast<size_t>(end - p)));
    if (q == nullptr)
        return false;
    r.op.assign(p, q);
    return true;
}

} // namespace

auto listLaunches(const juce::File &logsDir) -> std::vector<Launch> {
    std::vector<Launch> out;
    // Current layout: pairs with the same stamp directly in Logs/.
    std::map<juce::String, Launch> byStamp;
    for (const auto &f: logsDir.findChildFiles(juce::File::findFiles, false, "oplog_*.jsonl.gz")) {
        const auto stamp = stampOf(f, "oplog_", ".jsonl.gz");
        byStamp[stamp].stamp = stamp;
        byStamp[stamp].oplog = f;
    }
    for (const auto &f: logsDir.findChildFiles(juce::File::findFiles, false, "session_*.log")) {
        const auto stamp = stampOf(f, "session_", ".log");
        byStamp[stamp].stamp = stamp;
        byStamp[stamp].sessionLog = f;
    }
    for (auto &[stamp, l]: byStamp) {
        l.time = timeOfStamp(stamp);
        out.push_back(l);
    }
    // Before Nov 2026: Logs/oplogs and Logs/sessions, each with its own stamp a moment apart.
    std::vector<Launch> old;
    for (const auto &f: logsDir.getChildFile("sessions").findChildFiles(juce::File::findFiles, false, "session_*.log*")) {
        const auto name = f.getFileName();
        const auto stamp = stampOf(f, "session_", name.endsWith(".gz") ? ".log.gz" : ".log");
        old.push_back({stamp, {}, f, timeOfStamp(stamp)});
    }
    for (const auto &f: logsDir.getChildFile("oplogs").findChildFiles(juce::File::findFiles, false, "oplog_*.jsonl.gz")) {
        const auto stamp = stampOf(f, "oplog_", ".jsonl.gz");
        const auto t = timeOfStamp(stamp);
        Launch *nearest = nullptr;
        for (auto &l: old)
            if (! l.oplog.exists() && std::abs((l.time - t).inSeconds()) <= 5.0
                && (nearest == nullptr || std::abs((l.time - t).inSeconds()) < std::abs((nearest->time - t).inSeconds())))
                nearest = &l;
        if (nearest != nullptr)
            nearest->oplog = f;
        else
            old.push_back({stamp, f, {}, t});
    }
    out.insert(out.end(), old.begin(), old.end());
    std::sort(out.begin(), out.end(), [](const Launch &a, const Launch &b) { return a.time > b.time; });
    return out;
}

auto Log::load(const juce::File &file) -> std::unique_ptr<Log> {
    auto log = std::make_unique<Log>();
    std::unique_ptr<juce::InputStream> in = file.createInputStream();
    if (in == nullptr) {
        log->error = "cannot open " + file.getFullPathName();
        return log;
    }
    if (file.getFileName().endsWith(".gz"))
        in = std::make_unique<juce::GZIPDecompressorInputStream>(
            in.release(), true, juce::GZIPDecompressorInputStream::gzipFormat);
    // Read until the stream ends or fails (a crashed run's gzip stream has no trailer).
    std::vector<char> buffer(1 << 20);
    while (true) {
        const int n = in->read(buffer.data(), static_cast<int>(buffer.size()));
        if (n <= 0)
            break;
        log->text.append(buffer.data(), static_cast<size_t>(n));
    }
    if (log->text.empty()) {
        log->error = "nothing could be read from " + file.getFileName();
        return log;
    }
    log->index();
    return log;
}

auto Log::fromText(std::string text) -> std::unique_ptr<Log> {
    auto log = std::make_unique<Log>();
    log->text = std::move(text);
    log->index();
    return log;
}

auto Log::index() -> void {
    size_t at = 0;
    while (at < text.size()) {
        auto nl = text.find('\n', at);
        const bool last = nl == std::string::npos;
        if (last)
            nl = text.size();
        Record r;
        if (nl > at && text[nl - 1] == '}' && parseHeader(text.data() + at, text.data() + nl, r)) {
            r.offset = at;
            r.length = nl - at;
            recs.push_back(std::move(r));
        } else if (nl > at) {
            ++skipped;
        }
        at = nl + 1;
    }
    // Sessions: RUN_START .. RUN_STOP (or the next start, or the end).
    std::optional<Session> open;
    const auto close = [&](size_t lastIndex, const std::string &reason) {
        open->last = lastIndex;
        open->endMs = recs[lastIndex].t;
        open->stopReason = reason;
        sess.push_back(*open);
        open.reset();
    };
    for (size_t i = 0; i < recs.size(); ++i) {
        if (recs[i].op == "RUN_START") {
            if (open.has_value())
                close(i - 1, "");
            open = Session{stringField(line(i), "run"), i, i, recs[i].t, recs[i].t, ""};
        } else if (recs[i].op == "RUN_STOP" && open.has_value()) {
            close(i, stringField(line(i), "reason"));
        }
    }
    if (open.has_value() && ! recs.empty())
        close(recs.size() - 1, "");
}

auto Log::line(size_t index) const -> std::string {
    if (index >= recs.size())
        return {};
    return text.substr(recs[index].offset, recs[index].length);
}

auto Log::fields(size_t index) const -> juce::var {
    return juce::JSON::parse(juce::String(line(index)));
}

auto Log::indexAt(double ms) const -> size_t {
    const auto it = std::lower_bound(recs.begin(), recs.end(), ms, [](const Record &r, double v) { return r.t < v; });
    return static_cast<size_t>(it - recs.begin());
}

auto RtContext::tokens() const -> std::vector<int> {
    std::vector<int> out = prefix;
    for (const auto &item: items)
        out.insert(out.end(), item.tokens.begin(), item.tokens.end());
    return out;
}

auto sessionOf(const Log &log, size_t index) -> std::optional<size_t> {
    const auto &s = log.sessions();
    for (size_t i = 0; i < s.size(); ++i)
        if (index >= s[i].first && index <= s[i].last)
            return i;
    return std::nullopt;
}

namespace {

/** Index of the item starting at token position `pos` (or items.size() for the end), or -1. */
auto itemAt(const RtContext &c, int64_t pos) -> int64_t {
    int64_t p = static_cast<int64_t>(c.prefix.size());
    for (size_t i = 0; i < c.items.size(); ++i) {
        if (p == pos)
            return static_cast<int64_t>(i);
        p += static_cast<int64_t>(c.items[i].tokens.size());
    }
    return p == pos ? static_cast<int64_t>(c.items.size()) : -1;
}

auto contextFromCheckpoint(const std::string &model, const juce::var &part, size_t checkpoint) -> RtContext {
    RtContext c;
    c.model = model;
    c.base = static_cast<int>(part["base"]);
    c.fromCheckpoint = checkpoint;
    const auto tokens = intsOf(part["tokens"]);
    std::vector<std::pair<size_t, ContextItem>> items;
    if (const auto *a = part["items"].getArray())
        for (const auto &v: *a)
            items.push_back({static_cast<size_t>(static_cast<juce::int64>(v["pos"])),
                             {v["kind"].toString().toStdString(), v["id"].toString().toStdString(), {}}});
    std::sort(items.begin(), items.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    const size_t first = items.empty() ? tokens.size() : std::min(items.front().first, tokens.size());
    c.prefix.assign(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(first));
    for (size_t i = 0; i < items.size(); ++i) {
        const size_t from = std::min(items[i].first, tokens.size());
        const size_t to = i + 1 < items.size() ? std::min(items[i + 1].first, tokens.size()) : tokens.size();
        items[i].second.tokens.assign(tokens.begin() + static_cast<std::ptrdiff_t>(from),
                                      tokens.begin() + static_cast<std::ptrdiff_t>(std::max(from, to)));
        c.items.push_back(std::move(items[i].second));
    }
    return c;
}

} // namespace

auto stateAt(const Log &log, size_t index) -> State {
    State st;
    const auto &recs = log.records();
    if (recs.empty())
        return st;
    index = std::min(index, recs.size() - 1);
    const auto session = sessionOf(log, index);
    const size_t start = session.has_value() ? log.sessions()[*session].first : 0;

    // RT contexts: the last rt checkpoint in the session, then the context changes after it.
    std::optional<size_t> checkpoint;
    for (size_t i = index + 1; i-- > start;)
        if (recs[i].op == "CHECKPOINT" && log.line(i).find("\"part\":\"rt\"") != std::string::npos) {
            checkpoint = i;
            break;
        }
    if (checkpoint.has_value()) {
        st.rtCheckpointState = checkpoint;
        const auto f = log.fields(*checkpoint);
        if (auto *models = f["rt"].getDynamicObject())
            for (const auto &prop: models->getProperties()) {
                const auto model = prop.name.toString().toStdString();
                st.rt[model] = contextFromCheckpoint(model, prop.value, *checkpoint);
            }
        for (size_t i = *checkpoint + 1; i <= index; ++i) {
            const auto &op = recs[i].op;
            if (op != "RT_ADD" && op != "RT_REMOVE" && op != "RT_REWRITE" && op != "RT_REBASE")
                continue;
            const auto v = log.fields(i);
            const auto model = v["model"].toString().toStdString();
            auto it = st.rt.find(model);
            if (it == st.rt.end())
                continue;
            auto &c = it->second;
            if (op == "RT_ADD") {
                const auto at = itemAt(c, static_cast<juce::int64>(v["pos"]));
                ContextItem item{v["kind"].toString().toStdString(), v["id"].toString().toStdString(), intsOf(v["tokens"])};
                if (at < 0) {
                    c.exact = false;
                    c.items.push_back(std::move(item));
                } else {
                    c.items.insert(c.items.begin() + at, std::move(item));
                }
            } else if (op == "RT_REWRITE") {
                const auto at = itemAt(c, static_cast<juce::int64>(v["pos"]));
                if (at >= 0 && at < static_cast<int64_t>(c.items.size()))
                    c.items[static_cast<size_t>(at)].tokens = intsOf(v["new"]);
                else
                    c.exact = false;
            } else if (op == "RT_REMOVE") {
                std::vector<int64_t> at;
                if (const auto *a = v["items"].getArray())
                    for (const auto &x: *a)
                        at.push_back(itemAt(c, static_cast<juce::int64>(x["pos"])));
                std::sort(at.rbegin(), at.rend());
                for (const auto i2: at) {
                    if (i2 >= 0 && i2 < static_cast<int64_t>(c.items.size()))
                        c.items.erase(c.items.begin() + i2);
                    else
                        c.exact = false;
                }
            } else { // RT_REBASE: the oldest items go; the rest are re-encoded, known at the next checkpoint
                const auto dropped = static_cast<size_t>(static_cast<juce::int64>(v["dropped_items"]));
                c.items.erase(c.items.begin(), c.items.begin() + static_cast<std::ptrdiff_t>(std::min(dropped, c.items.size())));
                c.base = static_cast<int>(v["new_base"]);
                c.exact = false;
            }
        }
    }

    // OT input, vocsep, queues: scanned from the session's start.
    for (size_t i = start; i <= index; ++i) {
        const auto &op = recs[i].op;
        if (op == "OT_INPUT") {
            st.otInput = i;
            st.otDrawsSinceInput.clear();
        } else if (op == "OT_DRAW") {
            if (st.otInput.has_value())
                st.otDrawsSinceInput.push_back(i);
        } else if (op == "VOCSEP") {
            st.vocsep = i;
        } else if (op == "Q_PUSH" || op == "Q_POP") {
            const auto l = log.line(i);
            const auto queue = stringField(l, "queue");
            const auto item = stringField(l, "item");
            auto &q = st.queues[queue];
            if (op == "Q_PUSH") {
                q.push_back({item, i});
            } else {
                const auto it = std::find_if(q.begin(), q.end(), [&](const QueueItem &x) { return x.item == item; });
                if (it != q.end())
                    q.erase(it);
            }
        }
    }
    return st;
}

} // namespace OpLogReader
