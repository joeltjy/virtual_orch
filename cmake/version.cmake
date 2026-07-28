# Modified from https://www.mattkeeter.com/blog/2018-01-06-versioning/

execute_process(COMMAND bash "-c" "${CMAKE_CURRENT_BINARY_DIR}/tools/version.sh | tr -d '\n'"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE BUILD_REV
        ERROR_QUIET)

# Check whether we got any revision (which isn't
# always the case, e.g. when someone downloaded a zip
# file from Github instead of a checkout)
if ("${BUILD_REV}" STREQUAL "")
    set(BUILD_REV "unknown-error-determining-rev")
endif()

set(VERSION "const char* BUILD_REV=\"${BUILD_REV}\";")

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)
    file(READ ${CMAKE_CURRENT_BINARY_DIR}/version.cpp VERSION_)
else()
    set(VERSION_ "")
endif()

if (NOT "${VERSION}" STREQUAL "${VERSION_}")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/version.cpp "${VERSION}")
endif()
