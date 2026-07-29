cmake_minimum_required(VERSION 3.22)
project(VirtualOrch LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

# Enable LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

option(BUILD_TESTS "Build tests" OFF)

# AMD AOCL BLAS provider options
option(BUILD_BLAS_AOCL "Build AMD AOCL-BLIS as GGML BLAS provider (single thread)" OFF)
option(BUILD_BLAS_AOCL_MT "Build AMD AOCL-BLIS as GGML BLAS provider (multithread)" OFF)
set(AOCL_ADDITIONAL_OPTIONS "" CACHE STRING "Additional options for AOCL")
if (BUILD_BLAS_AOCL AND BUILD_BLAS_AOCL_MT)
    message(FATAL_ERROR "Cannot enable both BUILD_BLAS_AOCL and BUILD_BLAS_AOCL_MT")
endif()
if (BUILD_BLAS_AOCL OR BUILD_BLAS_AOCL_MT)
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        message(FATAL_ERROR "BUILD_BLAS_AOCL and BUILD_BLAS_AOCL_MT are not supported on macOS")
    endif()
endif()

# GGML options
option(ENABLE_GGML_BLAS "Enable GGML framework with BLAS CPU acceleration" OFF)
option(ENABLE_GGML_CPU "Enable GGML framework with CPU backend with SIMD optimizations" OFF)
option(ENABLE_GGML_CUDA "Enable GGML framework with NVIDIA GPU acceleration" OFF)
option(ENABLE_GGML_METAL "Enable GGML framework with Apple GPU acceleration" OFF)
option(ENABLE_GGML_SYCL "Enable GGML framework with Intel SYCL GPU acceleration" OFF)
option(ENABLE_GGML_VULKAN "Enable GGML framework with Vulkan GPU acceleration" OFF)
option(ENABLE_GGML_ZENDNN "Enable GGML framework with AMD ZenDNN CPU acceleration" OFF)
set(GGML_ADDITIONAL_OPTIONS "" CACHE STRING "Additional options for GGML backend")
set(GGML_BLAS_VENDOR "" CACHE STRING "BLAS vendor for GGML BLAS acceleration")
set(GGML_CPU_ARM_ARCH "" CACHE STRING "Architecture for GGML CPU backend on ARM platforms")
if ((BUILD_BLAS_AOCL OR BUILD_BLAS_AOCL_MT) AND NOT ENABLE_GGML_BLAS)
    message(WARNING "Build AOCL BLAS enabled but GGML BLAS acceleration is not enabled. AOCL BLAS will be built but unused.")
endif()
if (ENABLE_GGML_BLAS)
    message(STATUS "Enabling GGML CPU backend since GGML BLAS backend is enabled")
    set(ENABLE_GGML_CPU ON)
endif()
if (ENABLE_GGML_CUDA)
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        message(FATAL_ERROR "GGML CUDA backend is not supported on macOS")
    endif()
endif()
if (ENABLE_GGML_METAL)
    if (NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
        message(FATAL_ERROR "GGML Metal backend is only supported on macOS")
    endif()
endif()
if (ENABLE_GGML_SYCL)
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        message(FATAL_ERROR "GGML SYCL backend is not supported on macOS")
    endif()
endif()
if (ENABLE_GGML_ZENDNN)
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        message(FATAL_ERROR "GGML ZenDNN backend is not supported on macOS")
    endif()

    message(STATUS "Enabling GGML CPU backend since GGML ZenDNN backend is enabled")
    set(ENABLE_GGML_CPU ON)
endif()
if (ENABLE_GGML_BLAS AND ENABLE_GGML_ZENDNN)
    message(WARNING "GGML ZenDNN acceleration and GGML BLAS acceleration both enabled. Which one is used is undefined.")
endif()
if (ENABLE_GGML_CPU OR ENABLE_GGML_CUDA OR ENABLE_GGML_METAL OR ENABLE_GGML_SYCL OR ENABLE_GGML_VULKAN)
    set(ENABLE_GGML ON)

    if (NOT (CMAKE_SYSTEM_NAME MATCHES "Darwin" OR CMAKE_SYSTEM_NAME MATCHES "Linux"))
        message(FATAL_ERROR "Unsupported platform for GGML")
    endif()
endif()

# Torch options
option(ENABLE_TORCH_CPU "Enable Torch framework with CPU backend" OFF)
option(ENABLE_TORCH_CUDA "Enable Torch framework with NVIDIA GPU acceleration" OFF)
option(ENABLE_TORCH_MPS "Enable Torch framework with Apple GPU acceleration" OFF)
if (ENABLE_TORCH_CUDA)
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        message(FATAL_ERROR "Torch CUDA backend is not supported on macOS")
    endif()
endif()
if (ENABLE_TORCH_MPS)
    if (NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
        message(FATAL_ERROR "Torch MPS backend is only supported on macOS")
    endif()
endif()
if (ENABLE_TORCH_CPU OR ENABLE_TORCH_CUDA OR ENABLE_TORCH_MPS)
    set(ENABLE_TORCH ON)

    if ((DEFINED TORCH_INCLUDE AND NOT DEFINED TORCH_LIB_DIR) OR (NOT DEFINED TORCH_INCLUDE AND DEFINED TORCH_LIB_DIR))
        message(FATAL_ERROR "Please set both TORCH_INCLUDE and TORCH_LIB_DIR to use a local Torch installation. Set neither to download Torch automatically.")
    endif()

    if (NOT (CMAKE_SYSTEM_NAME MATCHES "Darwin" OR CMAKE_SYSTEM_NAME MATCHES "Linux"))
        message(FATAL_ERROR "Unsupported platform for Torch")
    endif()
endif()

# CoreML options
option(ENABLE_COREML "Enable CoreML framework for Apple acceleration" OFF)
if (ENABLE_COREML)
    if (NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
        message(FATAL_ERROR "Core ML backend is only supported on macOS")
    endif()

    if(NOT DEFINED COREML_MODEL_SOURCE)
        message(FATAL_ERROR "Please provide a CoreML model source file with COREML_MODEL_SOURCE")
    endif()
endif()

# ONNX options
option(ENABLE_ONNXRUNTIME_CPU "Enable ONNX Runtime with CPU backend" OFF)
option(ENABLE_ONNXRUNTIME_TENSORRT "Enable ONNX Runtime with NVIDIA TensorRT GPU acceleration" OFF)
option(ENABLE_ONNXRUNTIME_CUDA "Enable ONNX Runtime with NVIDIA CUDA GPU acceleration" OFF)
option(ONNXRUNTIME_BUILD_FROM_SOURCE "Build ONNX Runtime from source" OFF)
option(ONNXRUNTIME_BUILD_FROM_SOURCE_CCACHE "Enable CCache when building ONNX Runtime from source" OFF)
set(ONNXRUNTIME_BUILD_TYPE "Release" CACHE STRING "ONNX Runtime build type when building from source")
set(ONNXRUNTIME_VERSION "1.23.1" CACHE STRING "ONNX Runtime version to fetch or build")
if (ENABLE_ONNXRUNTIME_TENSORRT)
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        message(FATAL_ERROR "ONNX TensorRT backend is not supported on macOS")
    endif()
endif()
if (ENABLE_ONNXRUNTIME_CUDA)
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        message(FATAL_ERROR "ONNX CUDA backend is not supported on macOS")
    endif()
endif()
if (ENABLE_ONNXRUNTIME_CPU OR ENABLE_ONNXRUNTIME_TENSORRT OR ENABLE_ONNXRUNTIME_CUDA)
    set(ENABLE_ONNXRUNTIME ON)

    if ((DEFINED ONNXRUNTIME_INCLUDE OR DEFINED ONNXRUNTIME_LIB_DIR) AND ONNXRUNTIME_BUILD_FROM_SOURCE)
        message(FATAL_ERROR "Invalid combination of ONNXRUNTIME_INCLUDE, ONNXRUNTIME_LIB_DIR, and ONNXRUNTIME_BUILD_FROM_SOURCE options. Please set both ONNXRUNTIME_INCLUDE and ONNXRUNTIME_LIB_DIR together to use a local installation. Set ONNXRUNTIME_BUILD_FROM_SOURCE alone to build from source. Set neither to download binaries automatically.")
    elseif ((DEFINED ONNXRUNTIME_INCLUDE AND NOT DEFINED ONNXRUNTIME_LIB_DIR) OR (NOT DEFINED ONNXRUNTIME_INCLUDE AND DEFINED ONNXRUNTIME_LIB_DIR))
        message(FATAL_ERROR "Please set both ONNXRUNTIME_INCLUDE and ONNXRUNTIME_LIB_DIR to use a local installation. Use ONNXRUNTIME_BUILD_FROM_SOURCE to build instead. Set neither to download binaries automatically.")
    endif()

    if (NOT (CMAKE_SYSTEM_NAME MATCHES "Darwin" OR CMAKE_SYSTEM_NAME MATCHES "Linux"))
        message(FATAL_ERROR "Unsupported platform for ONNX Runtime")
    endif()
endif()

if (NOT (ENABLE_GGML OR ENABLE_TORCH OR ENABLE_COREML OR ENABLE_ONNXRUNTIME))
    message(FATAL_ERROR "Please enable at least one of the ML frameworks")
endif()


# Avoid DOWNLOAD_EXTRACT_TIMESTAMP warning
cmake_policy(SET CMP0135 NEW)

# Add custom CMake module path (for FindCUDNN)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

# Add CPM
include(cmake/CPM.cmake)

# See https://github.com/ggml-org/ggml/pull/1394
# NOTE: GGML incorrectly uses CMAKE_RUNTIME_OUTPUT_DIRECTORY instead of a per-target RUNTIME_OUTPUT_DIRECTORY
# TODO: When this gets pulled into a separate library these can be linked not in the app dir
if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/app/${PROJECT_NAME}_artefacts/$<CONFIG>)
endif()

if (BUILD_BLAS_AOCL OR BUILD_BLAS_AOCL_MT)
    set(BLAS_AOCL_OPTIONS "${BLAS_AOCL_ADDITIONAL_OPTIONS}")
    set(BLAS_AOCL_OPTIONS "${BLAS_AOCL_OPTIONS}" "BUILD_STATIC_LIBS OFF")
    set(BLAS_AOCL_OPTIONS "${BLAS_AOCL_OPTIONS}" "BUILD_SHARED_LIBS ON")
    set(BLAS_AOCL_OPTIONS "${BLAS_AOCL_OPTIONS}" "ENABLE_BLAS ON")
    set(BLAS_AOCL_OPTIONS "${BLAS_AOCL_OPTIONS}" "ENABLE_CBLAS ON")
    set(BLAS_AOCL_OPTIONS "${BLAS_AOCL_OPTIONS}" "BLIS_CONFIG_FAMILY amdzen")
    set(BLAS_AOCL_OPTIONS "${BLAS_AOCL_OPTIONS}" "ENABLE_TRSM_PREINVERSION ON")
    set(BLAS_AOCL_OPTIONS "${BLAS_AOCL_OPTIONS}" "ENABLE_ADDON aocl_gemm")
    if (CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(BLAS_AOCL_OPTIONS "${BLAS_AOCL_OPTIONS}" "COMPLEX_RETURN intel")
    else()
        set(BLAS_AOCL_OPTIONS "${BLAS_AOCL_OPTIONS}" "COMPLEX_RETURN default")
    endif()

    if (BUILD_BLAS_AOCL_MT)
        set(BLAS_AOCL_OPTIONS "${BLAS_AOCL_OPTIONS}" "ENABLE_THREADING openmp")
    else()
        set(BLAS_AOCL_OPTIONS "${BLAS_AOCL_OPTIONS}" "ENABLE_THREADING no")
    endif()

    CPMAddPackage(
            NAME blas
            GITHUB_REPOSITORY amd/blis
            GIT_TAG AOCL-Weekly-121225
            OPTIONS
                ${BLAS_AOCL_OPTIONS}
    )
endif()

# ###### #
#  GGML  #
# ###### #
if (ENABLE_GGML)
    add_definitions(-DENABLE_GGML)

    set(GGML_OPTIONS "${GGML_ADDITIONAL_OPTIONS}")
    set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_LTO ON")
    set(GGML_OPTIONS "${GGML_OPTIONS}" "BUILD_SHARED_LIBS ON") # Necessary for GGML_BACKEND_DL
    set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_BACKEND_DL ON") # Necessary for GGML_CPU_ALL_VARIANTS
    set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_NATIVE OFF") # Necessary for GGML_CPU_ALL_VARIANTS
    set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_OPENMP ON") # Parallelization from OpenMP
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin") # Handle MacOS bundle shared libraries
        # Since we don't know PROJECT_NAME in this file, we set the install dir to the build dir and
        # use an install task in the app CMakeLists.txt to copy the shared library into the app bundle
        # TODO(Perry): Fix when app becomes library
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_BACKEND_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}")
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_BACKEND_RELATIVE_TO_EXE_SEARCH_DIR ../Frameworks/")
    endif()

    if(ENABLE_GGML_BLAS)
        add_definitions(-DENABLE_GGML_BLAS)

        if (BUILD_BLAS_AOCL OR BUILD_BLAS_AOCL_MT)
            set(GGML_OPTIONS "${GGML_OPTIONS}" "BLAS_LIBRARIES AOCL::BLIS_shared")
            set(GGML_OPTIONS "${GGML_OPTIONS}" "BLAS_INCLUDE_DIRS ${blas_BINARY_DIR}/include/amdzen")
            if (BUILD_BLAS_AOCL_MT)
                set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_BLAS_VENDOR AOCL_mt")
                message(STATUS "GGML BLAS enabled for AOCL_mt from BUILD_BLAS_AOCL_MT option")
            elseif (BUILD_BLAS_AOCL)
                set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_BLAS_VENDOR AOCL")
                message(STATUS "GGML BLAS enabled for AOCL from BUILD_BLAS_AOCL option")
            endif()
        else()
            set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_BLAS_VENDOR ${GGML_BLAS_VENDOR}")
        endif()

        message(STATUS "GGML enabled with BLAS acceleration (vendor: ${GGML_BLAS_VENDOR})")
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_BLAS ON")
    else()
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_BLAS OFF")
    endif()

    if(ENABLE_GGML_CPU)
        add_definitions(-DENABLE_GGML_CPU)
        if (GGML_CPU_ARM_ARCH)
            set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_CPU_ARM_ARCH ${GGML_CPU_ARM_ARCH}")
        else()
            set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_CPU_ALL_VARIANTS ON")
        endif()

        message(STATUS "GGML enabled with CPU backend")
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_CPU ON")
    else()
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_CPU OFF")
    endif()

    if (ENABLE_GGML_CUDA)
        add_definitions(-DENABLE_GGML_CUDA)
        message(STATUS "GGML enabled with NVIDIA CUDA")
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_CUDA ON")
    else()
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_CUDA OFF")
    endif()

    if(ENABLE_GGML_METAL)
        add_definitions(-DENABLE_GGML_METAL)
        message(STATUS "GGML enabled with Apple Metal")
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_METAL ON")
    else()
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_METAL OFF")
    endif()

    if(ENABLE_GGML_SYCL)
        add_definitions(-DENABLE_GGML_SYCL)
        message(STATUS "GGML enabled with Intel SYCL")
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_SYCL ON")
    else()
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_SYCL OFF")
    endif()

    if(ENABLE_GGML_VULKAN)
        add_definitions(-DENABLE_GGML_VULKAN)
        message(STATUS "GGML enabled with Vulkan")
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_VULKAN ON")
    else()
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_VULKAN OFF")
    endif()

    if(ENABLE_GGML_ZENDNN)
        add_definitions(-DENABLE_GGML_ZENDNN)
        message(STATUS "GGML enabled with AND ZenDNN acceleration")
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_ZENDNN ON")
        # set(GGML_OPTIONS "${GGML_OPTIONS}" "ZENDNNL_MATMUL_ALGO 2") # TODO (Perry): Test algorithms
    else()
        set(GGML_OPTIONS "${GGML_OPTIONS}" "GGML_ZENDNN OFF")
    endif()

    # Download GGML
    CPMAddPackage(
            NAME GGML
            GITHUB_REPOSITORY mitmedialab/ggml
            GIT_TAG 2862091e4eda3577aafae895fe1e151e0c6739c2
            OPTIONS
                ${GGML_OPTIONS}
    )
    set(ML_FRAMEWORK_INCLUDE ${ML_FRAMEWORK_INCLUDE} ${GGML_SOURCE_DIR}/include)
    set(ML_FRAMEWORK_LIBRARY ${ML_FRAMEWORK_LIBRARY} "ggml")
endif()

# ####### #
#  TORCH  #
# ####### #
if(ENABLE_TORCH)
    add_definitions(-DENABLE_TORCH)

    # Set device for PyTorch
    # TODO (Perry): Build with multiple device support
    if (ENABLE_TORCH_CUDA)
        add_definitions(-DENABLE_TORCH_CUDA)
        message(STATUS "Torch enabled with CUDA")
        # TODO: Adapt to CUDA version found on system
        find_package(CUDAToolkit REQUIRED)
        set(TORCH_DEVICE "cu129")
    elseif(ENABLE_TORCH_MPS)
        add_definitions(-DENABLE_TORCH_CUDA)
        message(STATUS "Torch enabled with MPS")
    elseif(ENABLE_TORCH_CPU)
        add_definitions(-DENABLE_TORCH_CPUA)
        message(STATUS "Torch enabled with CPU")
        set(TORCH_DEVICE "cpu")
    endif()

    # Set platform for PyTorch
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(TORCH_PLATFORM "macos-arm64")
    elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(TORCH_PLATFORM "cxx11-abi-shared-with-deps")
    endif()

    # TODO (Perry): Build Torch option
    # TODO (Perry): Find Torch option
    if(DEFINED TORCH_INCLUDE AND DEFINED TORCH_LIB_DIR)
        message(STATUS "Using provided Torch include and lib directories")
    else ()
        message(STATUS "Fetching Torch")
        include(FetchContent)
        FetchContent_Declare(
                torch
                URL https://download.pytorch.org/libtorch/nightly/${TORCH_DEVICE}/libtorch-${TORCH_PLATFORM}-latest.zip
        )
        FetchContent_MakeAvailable(torch)
        set(TORCH_INCLUDE ${torch_SOURCE_DIR}/include CACHE PATH "Torch include directories")
        set(TORCH_LIB_DIR ${torch_SOURCE_DIR}/lib CACHE PATH "Torch library directories")
    endif()
    set(ML_FRAMEWORK_INCLUDE ${ML_FRAMEWORK_INCLUDE} ${TORCH_INCLUDE})
    set(ML_FRAMEWORK_LIB_DIR ${ML_FRAMEWORK_LIB_DIR} ${TORCH_LIB_DIR})
    set(ML_FRAMEWORK_LIBRARY ${ML_FRAMEWORK_LIBRARY} "torch_cpu" "c10") # Required libraries for PyTorch
endif()

# ######## #
#  COREML  #
# ######## #
if(ENABLE_COREML)
    add_definitions(-DENABLE_COREML)


    # Find the CoreML model source files
    # TODO: This should be compiled directly from the provided model files
    file(GLOB_RECURSE COREML_MODEL_SOURCE_FILES ${COREML_MODEL_SOURCE}/*.m)

    # Adding Objective-C++ support
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(ACCELERATE_FRAMEWORK Accelerate)

    # Find CoreML
    find_library(COREML_FRAMEWORK CoreML)

    set(ML_FRAMEWORK_INCLUDE ${ML_FRAMEWORK_INCLUDE} ${COREML_MODEL_SOURCE})
    set(ML_FRAMEWORK_LIBRARY ${ML_FRAMEWORK_LIBRARY} ${FOUNDATION_FRAMEWORK} ${ACCELERATE_FRAMEWORK} ${COREML_FRAMEWORK})
    set(ML_FRAMEWORK_SOURCE ${ML_FRAMEWORK_SOURCE} ${COREML_MODEL_SOURCE_FILES})
endif()

# ###### #
#  ONNX  #
# ###### #
if(ENABLE_ONNXRUNTIME) # ONNX RUNTIME
    add_definitions(-DENABLE_ONNXRUNTIME)

    if(ENABLE_ONNXRUNTIME_CPU)
        add_definitions(-DENABLE_ONNXRUNTIME_CPU)
        message(STATUS "ONNX Runtime enabled with CPU backend")
    endif()

    if(ENABLE_ONNXRUNTIME_TENSORRT)
        add_definitions(-DENABLE_ONNXRUNTIME_TENSORRT)
        message(STATUS "ONNX Runtime enabled with NVIDIA TensorRT backend")
    endif()

    if(ENABLE_ONNXRUNTIME_CUDA)
        add_definitions(-DENABLE_ONNXRUNTIME_CUDA)
        message(STATUS "ONNX Runtime enabled with NVIDIA CUDA backend")
    endif()

    if (ONNXRUNTIME_BUILD_FROM_SOURCE)
        # Set platform-dependent args to build ONNXRuntime
        if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
            set(ONNXRUNTIME_BUILD_ARGS "${ONNXRUNTIME_BUILD_ARGS}" " --cmake_extra_defines" "CMAKE_OSX_ARCHITECTURES=arm64")
            # TODO: Try --osx_arch arm64, --osx_arch arm64e, --osx_arch x86_64 instead
        endif()

        if (ENABLE_ONNXRUNTIME_TENSORRT OR ENABLE_ONNXRUNTIME_CUDA)
            find_package(CUDAToolkit REQUIRED)
            if (CUDAToolkit_FOUND)
                message(STATUS "CUDAToolkit version: ${CUDAToolkit_VERSION}")
                message(STATUS "CUDAToolkit target directory: ${CUDAToolkit_TARGET_DIR}")
                message(STATUS "CUDAToolkit bin directory: ${CUDAToolkit_BIN_DIR}")
                message(STATUS "CUDAToolkit NVCC executable: ${CUDAToolkit_NVCC_EXECUTABLE}")
                cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH ONNXRUNTIME_CUDAToolkit_ROOT_DIR)
                message(STATUS "CUDAToolkit implied root directory for ONNXRuntime: ${ONNXRUNTIME_CUDAToolkit_ROOT_DIR}")
            endif()

            # TODO (Perry): Find TensorRT or OSS TensorRT (version matched!) https://onnxruntime.ai/docs/build/eps.html#tensorrt
            # TODO (Perry): Tensor RT RTX https://onnxruntime.ai/docs/build/eps.html#nvidia-tensorrt-rtx

            find_package(CUDNN REQUIRED)
            if (CUDNN_FOUND)
                set(CUDNN_HOME ${CUDNN_INCLUDE_DIR})
                message(STATUS "cuDNN found: ${CUDNN_HOME}")
            else ()
                message(FATAL_ERROR "cuDNN not found. Please set the CUDNN_HOME environment variable manually.")
            endif ()

            set(ONNXRUNTIME_BUILD_ARGS "${ONNXRUNTIME_BUILD_ARGS}" " --use_cuda" "--cuda_home" "${ONNXRUNTIME_CUDAToolkit_ROOT_DIR}" "--cudnn_home" "${CUDNN_HOME}")
        endif()

        if (ONNXRUNTIME_BUILD_FROM_SOURCE_CCACHE)
           set(ONNXRUNTIME_BUILD_ARGS "${ONNXRUNTIME_BUILD_ARGS}" " --use_cache")
        endif()

        # Download ONNX Runtime source
        CPMAddPackage(
                NAME onnxruntime
                GITHUB_REPOSITORY microsoft/onnxruntime
                VERSION ${ONNXRUNTIME_VERSION}
        )

        # Build ONNX Runtime using .sh script
        execute_process(
            COMMAND ${onnxruntime_SOURCE_DIR}/build.sh --config ${ONNX_BUILD_CONFIG} --build_shared_lib --parallel --nvcc_threads -1 --compile_no_warning_as_error --skip_submodule_sync --skip_tests --allow_running_as_root ${ONNXRUNTIME_BUILD_ARGS} ${TOOLCHAIN_ONNXRUNTIME_BUILD_ARGS}
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Failed to build ONNX Runtime")
        endif()

        set(
          ONNXRUNTIME_INCLUDE
          ${onnxruntime_SOURCE_DIR}/include/onnxruntime              # Allow for other providers (core subdir used in pre-built for providers)
          ${onnxruntime_SOURCE_DIR}/include/onnxruntime/core/session # Flatten session since no subdir for pre-built package
          CACHE PATH "ONNX Runtime include directories"
        )
        if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
          set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/build/MacOS/${ONNX_BUILD_CONFIG} CACHE PATH "ONNX Runtime library directories")
        elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
          set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/build/Linux/${ONNX_BUILD_CONFIG} CACHE PATH "ONNX Runtime library directories")
        endif ()
    elseif(DEFINED ONNXRUNTIME_INCLUDE AND DEFINED ONNXRUNTIME_LIB_DIR)
        set(
            ONNXRUNTIME_INCLUDE
            ${ONNXRUNTIME_INCLUDE}                          # Pre-built package
            ${ONNXRUNTIME_INCLUDE}/onnxruntime              # Linux local install to /usr/local
            ${ONNXRUNTIME_INCLUDE}/onnxruntime/core/session # Windows local install or source
            CACHE PATH "ONNX Runtime include directories"
        )
    else()
        include(FetchContent)
        if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
          set(ONNXRUNTIME_DOWNLOAD_MODIFIER "osx-arm64")
        elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
          set(ONNXRUNTIME_DOWNLOAD_MODIFIER "linux-aarch64")
        else()
          set(ONNXRUNTIME_DOWNLOAD_MODIFIER "linux-x64-gpu")
        endif()
        FetchContent_Declare(
            onnxruntime
            URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${ONNXRUNTIME_DOWNLOAD_MODIFIER}-${ONNXRUNTIME_VERSION}.tgz
        )
        FetchContent_MakeAvailable(onnxruntime)
        set(ONNXRUNTIME_INCLUDE ${onnxruntime_SOURCE_DIR}/include CACHE PATH "ONNX Runtime include directories")
        set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib CACHE PATH "ONNX Runtime library directories")
    endif()

    set(ML_FRAMEWORK_INCLUDE ${ML_FRAMEWORK_INCLUDE} ${ONNXRUNTIME_INCLUDE})
    set(ML_FRAMEWORK_LIB_DIR ${ML_FRAMEWORK_LIB_DIR} ${ONNXRUNTIME_LIB_DIR})
    set(ML_FRAMEWORK_LIBRARY ${ML_FRAMEWORK_LIBRARY} "onnxruntime")
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION             ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
        INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_INCLUDE}
    )
    set_target_properties("onnxruntime" PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:onnxruntime,INTERFACE_INCLUDE_DIRECTORIES>
    )
endif()

# Download JUCE
CPMAddPackage(
    NAME JUCE
    GITHUB_REPOSITORY juce-framework/JUCE
    GIT_TAG 8.0.12
    VERSION 8.0.12
)

# Find Freetype
find_package(Freetype REQUIRED)
if (NOT FREETYPE_FOUND)
    message(FATAL_ERROR "Freetype not found")
endif()


# Show all errors
if (MSVC)
    add_compile_options(/Wall /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(app)
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
