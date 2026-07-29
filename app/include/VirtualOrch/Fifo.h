#pragma once

#include <JuceHeader.h>

template<typename T>
struct Fifo {
    bool push(const T &t) {
        if (auto write = fifo.write(1); write.blockSize1 > 0) {
            queue[write.startIndex1] = t;
            return true;
        }
        return false;
    }

    bool pull(T &t) {
        if (auto read = fifo.read(1); read.blockSize1 > 0) {
            t = queue[read.startIndex1];
            return true;
        }
        return false;
    }

    int getNumAvailableForReading() const {
        return fifo.getNumReady();
    }

private:
    static constexpr int Capacity = 120;
    std::array<T, Capacity> queue = {};
    juce::AbstractFifo fifo{Capacity};
};

template<typename T>
struct CircularFifo {
    void push(const T &t) {
        // If the queue is full, we need to advance the read pointer
        // to make room for the new item
        if (fifo.getFreeSpace() == 0) {
            // Advance read pointer by 1 to "consume" the oldest item
            auto read = fifo.read(1);
            // We don't actually need to do anything with the data,
            // just advance the pointer
        }

        // Now we're guaranteed to have space for 1 item
        auto write = fifo.write(1);
        queue[write.startIndex1] = t;
    }

    bool pull(T &t) {
        if (auto read = fifo.read(1); read.blockSize1 > 0) {
            t = queue[read.startIndex1];
            return true;
        }
        return false;
    }

    int getNumAvailableForReading() const {
        return fifo.getNumReady();
    }

private:
    static constexpr int Capacity = 120;
    std::array<T, Capacity> queue = {};
    juce::AbstractFifo fifo{Capacity};
};
