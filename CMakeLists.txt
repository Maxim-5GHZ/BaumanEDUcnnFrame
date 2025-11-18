cmake_minimum_required(VERSION 3.10)
project(CNNFrame)

# Set C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the subdirectory for the CNN library
add_subdirectory(edunet)

# Find ncurses for snake visualization
find_package(Curses REQUIRED)

# Define the executable for the demonstration model
add_executable(cnn_demo "demonstration model/main.cpp" "demonstration model/mnist_app.cpp" "demonstration model/snake.hpp")

# Link the demonstration model executable against the CNN library and ncurses
target_link_libraries(cnn_demo PRIVATE cnn_lib ${CURSES_LIBRARIES})

# Include directories for the demonstration model
target_include_directories(cnn_demo PRIVATE "demonstration model" "${CMAKE_CURRENT_SOURCE_DIR}/edunet")

# Installation (optional)
install(TARGETS cnn_demo
    RUNTIME DESTINATION bin
)

# Enable testing with CTest
enable_testing()

# Example of adding a test
# add_test(NAME my_test COMMAND cnn_demo)
