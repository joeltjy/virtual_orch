#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

/**
 * gtest_main, but the process ends without the exit handlers. A binary that used a CUDA session
 * aborted at exit about half the time ("corrupted double-linked list", detected inside libcuda's
 * own teardown) after every test had passed: AddressSanitizer finds nothing in our code, and with
 * CUDA hidden it never happens. The results are printed by then; nothing after them is needed.
 */
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    std::fflush(nullptr);
    std::_Exit(result);
}
