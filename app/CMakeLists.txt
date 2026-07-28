cmake_minimum_required(VERSION 4.2)

# TODO (Perry): redo version method
project(JordanAI VERSION 0.0.1)

# Add JUCE target
juce_add_gui_app(${PROJECT_NAME}
    COMPANY_NAME "MITMediaLab"
    PRODUCT_NAME "JordanAI"
    NEEDS_MIDI_INPUT TRUE
    ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/resources/icon.png"
)

# Generate JUCE header
juce_generate_juce_header(${PROJECT_NAME})

# Copy lib dir into bundle resources (MacOS)
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(TARGET_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_artefacts/$<CONFIG>/${PROJECT_NAME}.app")
    set(TARGET_RESOURCES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    install(CODE "
        include(BundleUtilities)
        set(BU_CHMOD_BUNDLE_ITEMS TRUE)
        set(TARGET_BUNDLE \"${TARGET_BUNDLE}\")
        set(TARGET_RESOURCES \"${TARGET_RESOURCES}\")

        set(framework_path \"\${TARGET_BUNDLE}/Contents/Frameworks\")
        message(\"Framework path: \${framework_path}\")

        cmake_path(GET TARGET_RESOURCES PARENT_PATH folder_parent_path)
        file(GLOB_RECURSE resources_relative_so RELATIVE \"\${folder_parent_path}\" \"\${TARGET_RESOURCES}/*.so\")
        file(GLOB_RECURSE resources_relative_dylib RELATIVE \"\${folder_parent_path}\" \"\${TARGET_RESOURCES}/*.dylib\")
        list (APPEND resources_relative \${resources_relative_so})
        list (APPEND resources_relative \${resources_relative_dylib})
        list(FILTER resources_relative EXCLUDE REGEX \"\${TARGET_BUNDLE}/.*\" )

        foreach(file IN LISTS resources_relative)
            message(\"Processing resource file: \${file}\")
            cmake_path(GET file FILENAME filename)
            set(bundle_lib \"\${framework_path}/\${filename}\")
            copy_resolved_item_into_bundle(\"\${folder_parent_path}/\${file}\" \"\${bundle_lib}\")
            list(APPEND fixup_libs \"\${bundle_lib}\")
        endforeach()

        fixup_bundle(
            \"\${TARGET_BUNDLE}\"
            \"\${fixup_libs}\"
            \"\${TARGET_RESOURCES}\"
        )
    ")
    # NOTE: Can't use CPack for signing because JUCE doesn't use install steps for bundle and relies on generator for paths
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    install(
        TARGETS ${PROJECT_NAME}
        DESTINATION bin
        COMPONENT application
    )


    # From https://cmake.org/cmake/help/v4.2/cpack_gen/appimage.html
    set(TARGET_RESOURCES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    install(
        CODE "
            set(TARGET_RESOURCES \"${TARGET_RESOURCES}\")

            message(\"Copying runtime dependencies for JordanAI...\")
            cmake_path(GET TARGET_RESOURCES PARENT_PATH folder_parent_path)
            file(GLOB_RECURSE resources \"\${folder_parent_path}\" \"\${TARGET_RESOURCES}/*.so\")
            message(\"Processing resource file: \$<TARGET_FILE:JordanAI>\")
            foreach(file IN LISTS resources)
                message(\"Processing resource file: \${file}\")
            endforeach()
            file(GET_RUNTIME_DEPENDENCIES
                EXECUTABLES \$<TARGET_FILE:JordanAI>
                LIBRARIES \${resources}
                RESOLVED_DEPENDENCIES_VAR resolved_deps
                PRE_EXCLUDE_REGEXES libcuda\\..*  # Exclude NVIDIA GPU driver
                DIRECTORIES /usr/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu/lib  # CMake does not check this dir even if root in CMAKE_FIND_ROOT_PATH
            )

            foreach(dep \${resolved_deps})
                # copy the symlink
                file(COPY \${dep} DESTINATION \${CMAKE_INSTALL_PREFIX}/lib)

                # Resolve the real path of the dependency (follows symlinks)
                file(REAL_PATH \${dep} resolved_dep_path)

                # Copy the resolved file to the destination
                file(COPY \${resolved_dep_path} DESTINATION \${CMAKE_INSTALL_PREFIX}/lib)
            endforeach()
        "
        COMPONENT libraries
    )

    install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/resources/application.desktop"
        DESTINATION "share/applications"
        RENAME "${PROJECT_NAME}.desktop"
        COMPONENT application
    )
    install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/resources/icon.png"
        DESTINATION "share/icons/hicolor/512x512/apps"
        RENAME "${PROJECT_NAME}.png"
        COMPONENT application
    )

    set(CPACK_COMPONENTS_ALL application libraries)
    set(CPACK_COMPONENT_APPLICATION_DEPENDS libraries)
    set(CPACK_COMPONENT_APPLICATION_DESCRIPTION "The ${PROJECT_NAME} application executable and resources")
    set(CPACK_COMPONENT_APPLICATION_DISPLAY_NAME "${PROJECT_NAME} Application")
    set(CPACK_COMPONENT_LIBRARIES_DESCRIPTION "Shared libraries required for ${PROJECT_NAME} to run")
    set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "${PROJECT_NAME} Libraries")
    set(CPACK_PACKAGING_INSTALL_PREFIX /usr)
    set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
    set(CPACK_PACKAGE_ICON "${PROJECT_NAME}.png")

    set(CPACK_GENERATOR "AppImage")
    # set(CPACK_APPIMAGE_COMPONENT_INSTALL ON) # Not supported by AppImage generator
    set(CPACK_APPIMAGE_EXCLUDE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/resources/.appimageignore")
    set(CPACK_APPIMAGE_DESKTOP_FILE "${PROJECT_NAME}.desktop")
    set(CPACK_APPIMAGE_ICON "${PROJECT_NAME}.png")
    set(CPACK_APPIMAGE_MAIN_EXECUTABLE "${PROJECT_NAME}")
    set(CPACK_APPIMAGE_NO_APPSTREAM ON)
    include(CPack)
endif()

# Get all source files
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/JordanAI")
file(GLOB_RECURSE SourceFiles
    CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp"
        "${INCLUDE_DIR}/*.h"
)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    file(GLOB_RECURSE ObjectiveCSourceFiles
            CONFIGURE_DEPENDS
                "${CMAKE_CURRENT_SOURCE_DIR}/source/*.mm"
    )
    list(APPEND SourceFiles ${ObjectiveCSourceFiles})

    # Retrieve all Objective-C source files
    get_target_property(OBJC_SOURCES ${PROJECT_NAME} SOURCES)
    list(FILTER OBJC_SOURCES INCLUDE REGEX "\\.(m|mm)$")
    # Enable ARC for Objective-C/C++ files
    set_source_files_properties(${OBJC_SOURCES} PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

# # Modified from https://www.mattkeeter.com/blog/2018-01-06-versioning/
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../tools/version.sh ${CMAKE_CURRENT_BINARY_DIR}/tools/version.sh COPYONLY)
ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
           ${CMAKE_CURRENT_BINARY_DIR}/_version.cpp # intentional dummy file so always runs
    COMMAND ${CMAKE_COMMAND} -P
            ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/version.cmake
)

target_sources(${PROJECT_NAME}
    PRIVATE
        ${SourceFiles}
        ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
        ${ML_FRAMEWORK_SOURCE}
)

target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_APPLICATION_NAME_STRING="$<TARGET_PROPERTY:${PROJECT_NAME},JUCE_PRODUCT_NAME>"
        JUCE_APPLICATION_VERSION_STRING="$<TARGET_PROPERTY:${PROJECT_NAME},JUCE_VERSION>"
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        include
    SYSTEM PRIVATE
        ${ML_FRAMEWORK_INCLUDE}
        ${FREETYPE_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        juce::juce_audio_utils
        juce::juce_osc
        ${ML_FRAMEWORK_LIBRARY}
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

target_link_directories(${PROJECT_NAME}
    PRIVATE
        ${ML_FRAMEWORK_LIB_DIR}
)

# TODO (Perry): determine if necessary
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "" FILES ${SourceFiles})

# Create a static library for JordanAI
# add_library(JordanAILibrary
#     STATIC
#         ${SourceFiles}
#         ${ML_FRAMEWORK_SOURCE}
# )
# # Use the app's JuceHeader.h
# get_target_property(juce_library_code ${PROJECT_NAME} JUCE_GENERATED_SOURCES_DIRECTORY)
# target_include_directories(JordanAILibrary
#     PUBLIC
#         include
#         ${ML_FRAMEWORK_INCLUDE}
#         ${juce_library_code}
#         ${LOMSE_SOURCE_DIR}/include
#         ${LOMSE_SOURCE_DIR}/src/agg/include
#         ${LOMSE_BINARY_DIR}
#         ${FREETYPE_INCLUDE_DIRS}
#         ${CHORDNAMER_BINARY_DIR}
#         ${CUSTOMMATPLOT_SOURCE_DIR}/include/include
#         ${CUSTOMMATPLOT_BINARY_DIR}/include/include
# )
# target_link_directories(JordanAILibrary PUBLIC ${ML_FRAMEWORK_LIB_DIR})
# target_link_libraries(JordanAILibrary
#     PRIVATE
#         juce::juce_audio_utils
#         juce::juce_osc
#         ${ML_FRAMEWORK_LIBRARY}
#         lomse-shared
#         chordnamer
#         cmp_plot
#     PUBLIC
#         juce::juce_recommended_config_flags
#         juce::juce_recommended_lto_flags
#         juce::juce_recommended_warning_flags
# )
# Prevent GTK and curl dependencies
# target_compile_definitions(JordanAILibrary PRIVATE
#   JUCE_WEB_BROWSER=0
#   JUCE_USE_CURL=0)
