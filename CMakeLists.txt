cmake_minimum_required(VERSION 3.10)

# Project Name
project(HNSWLibServer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the executable
add_executable(server src/server.cpp)

# Include directories for external libraries
include_directories(external/crow/include)
include_directories(external/hnswlib)
include_directories(external/json/single_include)
include_directories(external/asio/asio/include)

# Link external libraries if needed
target_link_libraries(server pthread)

# Optional: Set output directory
set_target_properties(server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Optional: Make the build verbose
set(CMAKE_VERBOSE_MAKEFILE ON)