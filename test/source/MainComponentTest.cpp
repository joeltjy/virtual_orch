#include <gtest/gtest.h>
#include <VirtualOrch/MainComponent.h>

namespace virtual_orch_test {
    TEST(MainComponent, CanInstantiate) {

        // This lets us use JUCE's MessageManager without leaking.
        auto gui = juce::ScopedJuceInitialiser_GUI {};

        MainComponent mainComponent;
        ASSERT_TRUE(true);
    }
}
