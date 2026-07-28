set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE arm64)

set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc-15)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++-15)
set(CMAKE_FORTRAN_COMPILER /usr/bin/aarch64-linux-gnu-gfortran-15)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/aarch64-linux-gnu-gcc-15)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/include)
