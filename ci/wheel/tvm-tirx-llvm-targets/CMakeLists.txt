# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
cmake_minimum_required(VERSION 3.26)
project(tvm_tirx_llvm_targets LANGUAGES C CXX)

# Build the all-target library and install only its component. The matching
# runtime/compiler comes from the main distribution dependency.
set(TVM_TIRX_BUILD_PYTHON_MODULE OFF CACHE BOOL "Build Python package artifacts" FORCE)
set(TVM_TIRX_LLVM_NATIVE_ONLY OFF CACHE BOOL "Build only the native LLVM target" FORCE)
set(TVM_TIRX_BUILD_TESTS OFF CACHE BOOL "Build C++ tests")
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "Library installation directory" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../.." tvm-tirx)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../../LICENSE"
              "${CMAKE_CURRENT_SOURCE_DIR}/../../../NOTICE"
  DESTINATION . COMPONENT tvm_tirx_llvm_backend)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../../licenses/"
  DESTINATION licenses COMPONENT tvm_tirx_llvm_backend)
