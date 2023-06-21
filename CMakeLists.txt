cmake_minimum_required(VERSION 3.15)
project(SyclCPLX LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(cmake/CPM.cmake)

option(SYCL_CPLX_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(SYCL_CPLX_FAST_MATH "Enable fast math (not all tests will pass)" OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  add_compile_options(-fsycl -O2)
  add_compile_definitions(SYCL_CPLX_TEST_POW_WORKAROUND)
  add_link_options(-fsycl)
  if(NOT SYCL_CPLX_FAST_MATH)
    message(STATUS "${PROJECT_NAME}: using precise fp model for intel llvm compiler")
    add_compile_options(-fp-model=precise)
  else()
    message(STATUS "${PROJECT_NAME}: using fast fp mode for intel llvm compiler")
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  message(STATUS "${PROJECT_NAME}: building tests")

  # Setup Catch2
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/vendor/Catch2/CMakeLists.txt")
      message(FATAL_ERROR"The git submodule vendor/Catch2 is missing.\nTry running `git submodule update --init`.")
  endif()

  add_subdirectory(vendor/Catch2)

  # Load catch2 cmake module
  include(Catch)
  add_subdirectory(tests)
endif()

if (SYCL_CPLX_BUILD_BENCHMARKS)
  message(STATUS "${PROJECT_NAME}: building benchmarks")
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
