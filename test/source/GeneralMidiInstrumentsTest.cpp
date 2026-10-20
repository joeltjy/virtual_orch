#include <gtest/gtest.h>

#include <VirtualOrch/GeneralMidiInstruments.h>

namespace virtual_orch_test {

TEST(GeneralMidiInstruments, NameForIdKnownPrograms) {
    EXPECT_EQ(GeneralMidiInstruments::nameForId(0), "Acoustic Grand");
    EXPECT_EQ(GeneralMidiInstruments::nameForId(127), "Gunshot");
}

TEST(GeneralMidiInstruments, NameForIdOutOfRange) {
    EXPECT_EQ(GeneralMidiInstruments::nameForId(-1), "Instrument -1");
    EXPECT_EQ(GeneralMidiInstruments::nameForId(128), "Instrument 128");
}

} // namespace virtual_orch_test
