# Find necessary packages
find_package(Threads REQUIRED)
find_package(Curses REQUIRED)

# Explicitly list source files for the application
set(APP_SOURCES
    main.cpp
    mnist_app.cpp
)

# Create the executable
add_executable(cnn_suite ${APP_SOURCES})

# Link the executable against the library and other dependencies
target_link_libraries(cnn_suite PRIVATE cnn_lib Threads::Threads ${CURSES_LIBRARIES})

# Add a custom command to copy the mnist directory after build
add_custom_command(TARGET cnn_suite POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/mnist ${CMAKE_CURRENT_BINARY_DIR}/mnist
    COMMENT "Copying MNIST data to build directory"
)

# --- Installation ---
# Install the executable to the 'bin' directory relative to the build root
install(TARGETS cnn_suite RUNTIME DESTINATION bin)
