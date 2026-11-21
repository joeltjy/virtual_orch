cmake_minimum_required(VERSION 3.22)
project(OplogReplay VERSION 0.0.1)

# oplog_replay: replay recorded session logs with the real models and write operation logs.
juce_add_console_app(OplogReplay PRODUCT_NAME "oplog_replay")
target_sources(OplogReplay PRIVATE oplog_replay/Main.cpp)
target_include_directories(OplogReplay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../app/include ${ONNXRUNTIME_INCLUDE})
target_link_directories(OplogReplay PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(OplogReplay
    PRIVATE
        juce::juce_audio_utils
        juce::juce_osc
        onnxruntime
        VirtualOrchLibrary
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)
target_compile_definitions(OplogReplay PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
