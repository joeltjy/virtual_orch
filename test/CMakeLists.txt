cmake_minimum_required(VERSION 3.22)
project(VirtualOrchTest VERSION 0.0.1)

enable_testing()

juce_add_console_app(${PROJECT_NAME}
        PRODUCT_NAME "virtual-orch Test Runner"
)

file(GLOB_RECURSE TestSourceFiles CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")

target_sources(${PROJECT_NAME}
    PRIVATE
        ${TestSourceFiles}
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${GOOGLETEST_SOURCE_DIR}/googletest/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../app/include
        ${ONNXRUNTIME_INCLUDE}
)
target_link_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_LIB_DIR})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        juce::juce_audio_utils
        juce::juce_osc
        GTest::gtest_main
        onnxruntime
        VirtualOrchLibrary
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Prevent GTK and curl dependencies
target_compile_definitions(${PROJECT_NAME} PRIVATE
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME} DISCOVERY_MODE)
