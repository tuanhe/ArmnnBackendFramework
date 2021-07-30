#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

AUX_SOURCE_DIRECTORY(. armnnCustomBackend_sources_files)
list(APPEND armnnCustomBackend_sources ${armnnCustomBackend_sources_files})

add_library(armnnCustomBackend OBJECT ${armnnCustomBackend_sources})
target_include_directories(armnnCustomBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnCustomBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnCustomBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnCustomBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnCustomBackend PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)

add_subdirectory(workloads)

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()
