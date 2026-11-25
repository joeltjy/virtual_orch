cmake_minimum_required(VERSION 3.22)

project(VirtualOrch VERSION 0.0.1)

# Add JUCE target
juce_add_gui_app(${PROJECT_NAME}
    COMPANY_NAME "MITMediaLab"
    PRODUCT_NAME "virtual-orch"
    NEEDS_MIDI_INPUT TRUE
    CAMERA_PERMISSION_ENABLED TRUE
    CAMERA_PERMISSION_TEXT "Virtual Orchestra uses a camera for conducting gestures and ArUco markers."
    ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/resources/icon.png"
)

set(virtual_orch_web_browser 0)
if(APPLE)
    set(virtual_orch_web_browser 1)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(virtual_orch_webkit QUIET IMPORTED_TARGET webkit2gtk-4.0 gtk+-x11-3.0)
        if(virtual_orch_webkit_FOUND)
            set(virtual_orch_web_browser 1)
            target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::virtual_orch_webkit)
        endif()
    endif()
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE
    JUCE_WEB_BROWSER=${virtual_orch_web_browser}
    JUCE_USE_CURL=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        include
        ${ONNXRUNTIME_INCLUDE}
        ${CUSTOMMATPLOT_SOURCE_DIR}/include/include
        ${CUSTOMMATPLOT_BINARY_DIR}/include/include
)
target_link_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_LIB_DIR})

# Generate JUCE header
juce_generate_juce_header(${PROJECT_NAME})

juce_add_binary_data(VirtualOrchData
    HEADER_NAME VirtualOrchBinaryData.h
    NAMESPACE VirtualOrchBinaryData
    SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/resources/general_midi_instruments.csv"
        "${CMAKE_CURRENT_SOURCE_DIR}/resources/orch_seating_plan_no_perc.png"
        "${CMAKE_CURRENT_SOURCE_DIR}/resources/orch_seating_plan_sectors.png"
        "${CMAKE_CURRENT_SOURCE_DIR}/resources/NotoMusic-Regular.ttf"
)

# Chord names for the ScoreWidget: CppChordNamer vendored with fixes in third_party/chordnamer
# (see its README.md), without the upstream demo.
set(CHORDNAMER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/chordnamer")
add_library(chordnamer STATIC
    ${CHORDNAMER_DIR}/src/chord.cpp
    ${CHORDNAMER_DIR}/src/interval.cpp
    ${CHORDNAMER_DIR}/src/note.cpp
)
target_include_directories(chordnamer PUBLIC ${CHORDNAMER_DIR}/include)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        juce::juce_audio_utils
        juce::juce_osc
        onnxruntime
        cmp_plot
        VirtualOrchData
        chordnamer
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

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
        VirtualOrchData
        chordnamer
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
)
# No link-time optimisation here: this library is only for the tests and tools (the app compiles its
# own sources with it), and LTO's serial link made every test build take minutes.

target_compile_definitions(${PROJECT_NAME} PRIVATE
  VIRTUAL_ORCH_APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(VirtualOrchLibrary PRIVATE
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0
  VIRTUAL_ORCH_APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(cmp_plot PRIVATE
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0)


set(controller_source "${CMAKE_CURRENT_SOURCE_DIR}/../sympathetic")
if(APPLE)
    set(controller_resources "$<TARGET_BUNDLE_CONTENT_DIR:${PROJECT_NAME}>/Resources/sympathetic")
else()
    set(controller_resources "$<TARGET_FILE_DIR:${PROJECT_NAME}>/sympathetic")
endif()

add_custom_target(VirtualOrchControllerResources
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${controller_resources}/aruco"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${controller_source}/public"
            "${controller_resources}/public"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${controller_source}/sym_server.js"
            "${controller_resources}/sym_server.js"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${controller_source}/aruco/worker.py"
            "${controller_source}/aruco/marker_pdf.py" "${controller_source}/aruco/macos_camera.py"
            "${controller_source}/aruco/requirements.txt" "${controller_source}/aruco/hand_markers.py"
            "${controller_resources}/aruco"
    VERBATIM)
add_dependencies(${PROJECT_NAME} VirtualOrchControllerResources)
