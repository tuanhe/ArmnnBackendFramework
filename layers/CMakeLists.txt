#
# Copyright © 2021 tuanhe. All rights reserved.
# SPDX-License-Identifier: MIT
#
AUX_SOURCE_DIRECTORY(. armnnCustomBackendLayers_source_files)
list(APPEND armnnCustomBackendLayers_sources ${armnnCustomBackendLayers_source_files})

add_library(armnnCustomBackendLayers OBJECT ${armnnCustomBackendLayers_sources})
target_include_directories(armnnCustomBackendLayers PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnCustomBackendLayers PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnCustomBackendLayers PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnCustomBackendLayers PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnCustomBackendLayers PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
