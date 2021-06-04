#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnCustomBackendUnitTests_sources
    CustomCreateWorkloadTests.cpp
    CustomEndToEndTests.cpp
)

add_library(armnnCustomBackendUnitTests OBJECT ${armnnCustomBackendUnitTests_sources})
target_include_directories(armnnCustomBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnCustomBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnCustomBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnCustomBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnCustomBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
