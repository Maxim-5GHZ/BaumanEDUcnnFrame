cmake_minimum_required(VERSION 3.10)
project(BaumanEDUcnnFrame VERSION 0.1 LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler warnings
if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Collect top-level .cpp sources in the project root only (do NOT recurse into the build directory)
# Recursing caused CMake-generated compiler-id files in the build tree to be picked up and
# that introduced a second `main` (linker error). Use a non-recursive glob so only project
# source files in the repository root are used. If you keep sources in subfolders, add
# explicit lists or separate targets per directory.
file(GLOB SRC_FILES "${PROJECT_SOURCE_DIR}/*.cpp")

# If learnCNN.cpp is present, prefer to include it; otherwise SRC_FILES will contain all .cpp
if(NOT SRC_FILES)
  message(FATAL_ERROR "No .cpp source files found in project. Add source files or adjust CMakeLists.txt.")
endif()

add_executable(learnCNN ${SRC_FILES})

# Make headers available (project root)
target_include_directories(learnCNN PRIVATE "${PROJECT_SOURCE_DIR}")

# Link pthread on Unix if needed
if(UNIX)
  find_package(Threads REQUIRED)
  target_link_libraries(learnCNN PRIVATE Threads::Threads)
endif()

# Installation
install(TARGETS learnCNN RUNTIME DESTINATION bin)

# Helpful message
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Sources: ${SRC_FILES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
