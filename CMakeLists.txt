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

