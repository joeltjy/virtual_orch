#include <gtest/gtest.h>
#include <JordanAI/MainComponent.h>

namespace jordan_ai_test {
    TEST(MainComponent, CanInstantiate) {

        // This lets us use JUCE's MessageManager without leaking.
        auto gui = juce::ScopedJuceInitialiser_GUI {};

        MainComponent mainComponent;
        ASSERT_TRUE(true);
    }
}
