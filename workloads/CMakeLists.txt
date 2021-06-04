#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnCustomBackendWorkloads_sources
        CustomAdditionWorkload.cpp
        CustomAdditionWorkload.hpp
        CustomPreCompiledWorkload.cpp
        CustomPreCompiledWorkload.hpp
)

add_library(armnnCustomBackendWorkloads OBJECT ${armnnCustomBackendWorkloads_sources})
target_include_directories(armnnCustomBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnCustomBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnCustomBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnCustomBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnCustomBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
