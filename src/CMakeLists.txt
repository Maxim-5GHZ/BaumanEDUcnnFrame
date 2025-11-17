# Glob all source files for the library
file(GLOB LIB_SOURCES "*.cpp" "*.h")

# Create a static library from the sources
add_library(cnn_lib STATIC ${LIB_SOURCES})

# The public include directory for this library is its own source directory
target_include_directories(cnn_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
