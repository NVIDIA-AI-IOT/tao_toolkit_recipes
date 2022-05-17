# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 2.8.7)
set(PROJECT_NAME pointpillars)
EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCH )
message( STATUS "Architecture: ${ARCH}" )

find_package(CUDA REQUIRED)

set(CUDA_VERSION 11.3)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-${CUDA_VERSION})

SET(CMAKE_BUILD_TYPE "Release")
add_compile_options(-W)
add_compile_options(-std=c++11)

set(SMS 50 52 53 60 61 62 70 72 75 80 86)
foreach(sm ${SMS})
    set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
list(GET SMS -1 LATEST_SM)
set(GENCODE "${GENCODE} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin ${CMAKE_CXX_COMPILER}
    -Xcompiler -DWIN_INTERFACE_CUSTOM
    -Xcompiler -I/usr/${ARCH}-linux-gnu/include/
    -Xlinker -lsocket
    -Xlinker -rpath=/usr/lib/${ARCH}-linux-gnu/
    -Xlinker -rpath=/usr/${ARCH}-linux-gnu/lib/
    -Xlinker -L/usr/lib/${ARCH}-linux-gnu/
    -Xlinker -L/usr/${ARCH}-linux-gnu/lib/
)

set(TENSORRT_INCLUDE_DIRS /usr/include/${ARCH}-linux-gnu/)
set(TENSORRT_LIBRARY_DIRS /usr/lib/${ARCH}-linux-gnu/)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ../include/
)

link_directories(
    ${TENSORRT_LIBRARY_DIRS}
    /usr/lib/${ARCH}-linux-gnu
    /usr/${ARCH}-linux-gnu/lib/
)

file(GLOB_RECURSE SOURCE_FILES
    ../src/*.cu
    ../src/*.cpp
)

cuda_add_executable(${PROJECT_NAME} main.cpp ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME}
    libnvinfer.so
    libnvonnxparser.so
    libnvinfer_plugin.so
)
