# Find CUDNN
#  CUDNN_INCLUDE_DIRS - where to find cudnn.h, etc.
#  CUDNN_LIBRARIES - List of libraries when using cudnn.
#  CUDNN_FOUND - True if cudnn found.

find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS
        ENV CUDNN_HOME
        PATH_SUFFIXES include
)

find_library(CUDNN_LIBRARY cudnn
    HINTS
        ENV CUDNN_HOME
        PATH_SUFFIXES lib64 lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_LIBRARY CUDNN_INCLUDE_DIR)

if (CUDNN_FOUND)
    set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
    set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
else ()
    message(FATAL_ERROR "Could not find cuDNN. Please set the CUDNN_HOME environment variable.")
endif ()