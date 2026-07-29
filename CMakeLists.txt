cmake_minimum_required(VERSION 3.22)
project(VirtualOrch)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_ONNXRUNTIME "Build ONNX Runtime from source" OFF)
option(BUILD_ONNXRUNTIME_CCACHE "Use Ccache when building ONNX Runtime" OFF)
set(ONNXRUNTIME_VERSION 1.18.0 CACHE STRING "The version to use for ONNX Runtime")

set(ONNX_BUILD_CONFIG Release)

# Avoid DOWNLOAD_EXTRACT_TIMESTAMP warning
cmake_policy(SET CMP0135 NEW)

# Add custom CMake module path (for FindCUDNN)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

# Add CPM
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)
include(cmake/cpm.cmake)

if (BUILD_ONNXRUNTIME)
    # Set platform-dependent args to build ONNXRuntime
    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(MACOS TRUE)
        set(ONNXRUNTIME_BUILD_ARGS "--cmake_extra_defines" "CMAKE_OSX_ARCHITECTURES=arm64")
        # TODO: Try --osx_arch arm64, --osx_arch arm64e, --osx_arch x86_64 instead
    else ()
        find_package(CUDAToolkit REQUIRED)
        if (CUDAToolkit_FOUND)
            message(STATUS "CUDAToolkit found: ${CUDAToolkit_TARGET_DIR}")
        endif()
        find_package(CUDNN REQUIRED)
        if (CUDNN_FOUND)
            set(CUDNN_HOME ${CUDNN_INCLUDE_DIR})
            message(STATUS "cuDNN found: ${CUDNN_HOME}")
        else ()
            message(FATAL_ERROR "cuDNN not found. Please set the CUDNN_HOME environment variable manually.")
        endif ()

        set(ONNXRUNTIME_BUILD_ARGS "--use_cuda" "--cuda_home" "${CUDAToolkit_TARGET_DIR}" "--cudnn_home" "${CUDNN_HOME}")
    endif()
    if (BUILD_ONNXRUNTIME_CCACHE)
       set(ONNXRUNTIME_BUILD_ARGS "${ONNXRUNTIME_BUILD_ARGS}" "--use_cache")
    endif()

    # Download ONNX Runtime
    CPMAddPackage(
            NAME onnxruntime
            GITHUB_REPOSITORY microsoft/onnxruntime
            VERSION ${ONNXRUNTIME_VERSION}
            SOURCE_DIR ${LIB_DIR}/onnxruntime
    )

    # Build ONNX Runtime using .sh script
    execute_process(
        COMMAND ${onnxruntime_SOURCE_DIR}/build.sh --config ${ONNX_BUILD_CONFIG} --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --skip_tests --allow_running_as_root ${ONNXRUNTIME_BUILD_ARGS}
        RESULT_VARIABLE result
    )

    set(
      ONNXRUNTIME_INCLUDE
      ${onnxruntime_SOURCE_DIR}/include/onnxruntime              # Allow for other providers (core subdir used in pre-built for providers)
      ${onnxruntime_SOURCE_DIR}/include/onnxruntime/core/session # Flatten session since no subdir for pre-built package
      CACHE PATH "ONNX Runtime include directories"
    )
    if (${MACOS})
      set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/build/MacOS/${ONNX_BUILD_CONFIG} CACHE PATH "ONNX Runtime library directories")
    else ()
      set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/build/Linux/${ONNX_BUILD_CONFIG} CACHE PATH "ONNX Runtime library directories")
    endif ()

    if(result)
        message(FATAL_ERROR "Failed to build ONNX Runtime")
    endif()
elseif(DEFINED ONNXRUNTIME_INCLUDE AND DEFINED ONNXRUNTIME_LIB_DIR)
    set(
        ONNXRUNTIME_INCLUDE
        ${ONNXRUNTIME_INCLUDE}                          # Pre-built package
        ${ONNXRUNTIME_INCLUDE}/onnxruntime              # Linux local install to /usr/local
        ${ONNXRUNTIME_INCLUDE}/onnxruntime/core/session # Windows local install or source
        CACHE PATH "ONNX Runtime include directories"
    )
elseif(DEFINED ONNXRUNTIME_INCLUDE AND DEFINED ONNXRUNTIME_LIB_DIR)
    message(FATAL_ERROR "Please set both ONNXRUNTIME_INCLUDE and ONNXRUNTIME_LIB_DIR or neither to donwload automatically. Use BUILD_ONNXRUNTIME to build instead.")
else (DOWNLOAD_ONNXRUNTIME)
    include(FetchContent)
    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      set(ONNXRUNTIME_DOWNLOAD_MODIFIER "osx-arm64")
    else()
      set(ONNXRUNTIME_DOWNLOAD_MODIFIER "linux-x64-gpu-cuda12")
    endif()
    FetchContent_Declare(
        onnxruntime
        URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${ONNXRUNTIME_DOWNLOAD_MODIFIER}-${ONNXRUNTIME_VERSION}.tgz
        SOURCE_DIR ${LIB_DIR}/onnxruntime
    )
    FetchContent_MakeAvailable(onnxruntime)
    set(ONNXRUNTIME_INCLUDE ${onnxruntime_SOURCE_DIR}/include CACHE PATH "ONNX Runtime include directories")
    set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib CACHE PATH "ONNX Runtime library directories")
endif()

# Download JUCE
CPMAddPackage(
    NAME JUCE
    GITHUB_REPOSITORY juce-framework/JUCE
    GIT_TAG 7.0.12
    VERSION 7.0.12
    SOURCE_DIR ${LIB_DIR}/juce
)

# Download CustomMatPlot
CPMAddPackage(
        NAME CUSTOMMATPLOT
        GITHUB_REPOSITORY lancelotblanchard/CustomMatPlot
        GIT_TAG 0.0.1
        VERSION 0.0.1
        SOURCE_DIR ${LIB_DIR}/custommatplot
)

# Download Google Test
CPMAddPackage(
        NAME GOOGLETEST
        GITHUB_REPOSITORY google/googletest
        VERSION 1.14.0
        SOURCE_DIR ${LIB_DIR}/googletest
        OPTIONS
        "INSTALL_GTEST OFF"
        "gtest_force_shared_crt ON"
)

enable_testing()

# Show all errors
if (MSVC)
    add_compile_options(/Wall /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(app)
add_subdirectory(test)
