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

# --- Dependencies ---
find_package(Threads REQUIRED)
find_package(Curses REQUIRED)

# --- Header files for include directories ---
# All headers are in the project root
include_directories(${PROJECT_SOURCE_DIR})


# --- Main Application Target ---
add_executable(cnn_suite main.cpp DQNAgent.cpp)

# Link against pthreads and ncurses
target_link_libraries(cnn_suite PRIVATE Threads::Threads ${CURSES_LIBRARIES})

# Installation
install(TARGETS cnn_suite RUNTIME DESTINATION bin)


# --- Helpful messages ---
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Found Curses (ncurses) library: ${CURSES_LIBRARIES}")
