include(${CMAKE_CURRENT_LIST_DIR}/tc-linux-aarch64-gnu-15.cmake)

# set(GGML_CPU_ARM_ARCH "armv9.2-a+sve2-bitperm+sve2-aes+sve2-sha3+sve2-sm4+memtag+profile")
set(CMAKE_CUDA_ARCHITECTURES "121")
set(TOOLCHAIN_ONNXRUNTIME_BUILD_ARGS "--cmake_extra_defines" "CMAKE_CUDA_ARCHITECTURES=121")

string(APPEND CMAKE_C_FLAGS_INIT " -mcpu=gb10")
string(APPEND CMAKE_CXX_FLAGS_INIT " -mcpu=gb10")
string(APPEND CMAKE_FORTRAN_FLAGS_INIT " -mcpu=gb10")
string(APPEND CMAKE_CUDA_FLAGS_INIT " -mcpu=gb10")
string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " -mcpu=gb10")
