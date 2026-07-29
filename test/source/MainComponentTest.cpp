#include <gtest/gtest.h>
#include <VirtualOrch/AppSession.h>
#include <VirtualOrch/ui/MainComponent.h>

namespace virtual_orch_test {
    TEST(MainComponent, CanInstantiate) {

        // This lets us use JUCE's MessageManager without leaking.
        auto gui = juce::ScopedJuceInitialiser_GUI {};

        AppSession session;
        MainComponent mainComponent(session);
        ASSERT_TRUE(true);
    }
}
