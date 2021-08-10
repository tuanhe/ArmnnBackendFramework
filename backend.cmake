#
# Copyright © tuanhe. All rights reserved.
# SPDX-License-Identifier: MIT
#
if(CUSTOM_SUPPORT)
    #set(CUSTOM_SDK_ROOT ${SET_YOUR_SDK_DIR})
    set(CUSTOM_SDK_ROOT ./)
    if(NOT DEFINED CUSTOM_SDK_ROOT)
        message(FATAL_ERROR  "CUSTOM_SDK_ROOT is not set while you enable CUSTOM_SUPPORT")  
    endif()

    set(HEADER_FILE Custom.hpp)
    # Add the support library
    find_path(SUPPORT_LIBRARY_INCLUDE_DIR 
              ${HEADER_FILE}
              HINTS ${CUSTOM_SDK_ROOT}/include)
    if(NOT CUSTOM_SUPPORT_LIBRARY)
        message(WARNING "Custom support head file (${HEADER_FILE}) not found")
    else()
        message(STATUS "Custom support head file located at: ${CUSTOM_SUPPORT_LIBRARY}")
        include_directories(${SUPPORT_LIBRARY_INCLUDE_DIR})
    endif()

    set(LIB libCustom.so)
    find_library(CUSTOM_SUPPORT_LIBRARY
                 NAMES ${LIB}
                 HINTS ${CUSTOM_SDK_ROOT}/lib)
    if(NOT CUSTOM_SUPPORT_LIBRARY)
        message(WARNING "Custom support library (${LIB}) not found")
    else()
        message(STATUS "Custom support library located at: ${CUSTOM_SUPPORT_LIBRARY}")
        link_libraries(${CUSTOM_SUPPORT_LIBRARY})
    endif()

    # Build the backend
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR})
    list(APPEND armnnLibraries armnnCustomBackend)
    list(APPEND armnnLibraries armnnCustomBackendLayers)
    list(APPEND armnnLibraries armnnCustomBackendWorkloads)

    if(BUILD_UNIT_TESTS)
        list(APPEND armnnUnitTestLibraries armnnCustomBackendUnitTests)
    endif()

endif()