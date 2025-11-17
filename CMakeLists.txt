cmake_minimum_required(VERSION 3.10)
project(BaumanEDUcnnFrame VERSION 0.3 LANGUAGES CXX)

# Set global C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add subdirectories for the library and the application
add_subdirectory(src)
add_subdirectory(app)
