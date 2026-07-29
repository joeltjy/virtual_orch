cmake_minimum_required(VERSION 3.22)

project(VirtualOrch VERSION 0.0.1)

# Add JUCE target
juce_add_gui_app(${PROJECT_NAME}
    COMPANY_NAME "MITMediaLab"
    PRODUCT_NAME "virtual-orch"
    NEEDS_MIDI_INPUT TRUE
    ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/resources/icon.png"
)

target_compile_definitions(${PROJECT_NAME} PUBLIC JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        include
        ${ONNXRUNTIME_INCLUDE}
        ${CUSTOMMATPLOT_SOURCE_DIR}/include/include
        ${CUSTOMMATPLOT_BINARY_DIR}/include/include
)
target_link_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_LIB_DIR})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        juce::juce_audio_utils
        juce::juce_osc
        onnxruntime
        cmp_plot
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Generate JUCE header
juce_generate_juce_header(${PROJECT_NAME})

# Get all source files
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/VirtualOrch")
file(GLOB_RECURSE SourceFiles CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp" "${INCLUDE_DIR}/*.h")

# # Modified from https://www.mattkeeter.com/blog/2018-01-06-versioning/
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../tools/version.sh ${CMAKE_CURRENT_BINARY_DIR}/tools/version.sh COPYONLY)
ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
           ${CMAKE_CURRENT_BINARY_DIR}/_version.cpp # intentional dummy file so always runs
    COMMAND ${CMAKE_COMMAND} -P
            ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/version.cmake)

target_sources(${PROJECT_NAME}
    PRIVATE
        ${SourceFiles}
        ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "" FILES ${SourceFiles})

# Create a static library for VirtualOrch
add_library(VirtualOrchLibrary
    STATIC
        ${SourceFiles}
)
# Use the app's JuceHeader.h
get_target_property(juce_library_code ${PROJECT_NAME} JUCE_GENERATED_SOURCES_DIRECTORY)
target_include_directories(VirtualOrchLibrary
    PUBLIC
        include
        ${ONNXRUNTIME_INCLUDE}
        ${juce_library_code}
        ${CUSTOMMATPLOT_SOURCE_DIR}/include/include
        ${CUSTOMMATPLOT_BINARY_DIR}/include/include
)
target_link_directories(VirtualOrchLibrary PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(VirtualOrchLibrary
    PRIVATE
        juce::juce_audio_utils
        juce::juce_osc
        onnxruntime
        cmp_plot
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Prevent GTK and curl dependencies
target_compile_definitions(${PROJECT_NAME} PRIVATE
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0)
target_compile_definitions(VirtualOrchLibrary PRIVATE
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0)
target_compile_definitions(cmp_plot PRIVATE
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0)
