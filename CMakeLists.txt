cmake_minimum_required(VERSION 3.15)
project(sandbox LANGUAGES CXX)

# Setup Catch2
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/vendor/Catch2/CMakeLists.txt")
    message(FATAL_ERROR "The git submodule vendor/Catch2 is missing.\nTry running `git submodule update --init`.")
endif()

add_subdirectory(vendor/Catch2)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Load catch2 cmake module
include(CTest)
include(Catch)

add_subdirectory(tests)

include(cmake/CPM.cmake)

if (SYCL_CPLX_BUILD_BENCHMARKS)
  CPMAddPackage(
    NAME benchmark
    GITHUB_REPOSITORY google/benchmark
    GIT_TAG v1.7.1
    VERSION v1.7.1
    GIT_SHALLOW TRUE
    OPTIONS
      "BENCHMARK_ENABLE_TESTING OFF"
      "BENCHMARK_ENABLE_INSTALL OFF"
      "BENCHMARK_ENABLE_WERROR OFF"
      "CMAKE_BUILD_TYPE Release"
  )
  add_subdirectory(benchmarks)
endif()
