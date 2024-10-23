cmake_minimum_required(VERSION 3.10)

# Project Name
project(HNSWLibServer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build types
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags for release
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffunction-sections -fdata-sections")

# Static linking
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static -Wl,--gc-sections")

# Enable LTO if supported
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
else()
    message(WARNING "LTO is not supported by the current compiler.")
endif()

# Define the executable
add_executable(server src/server.cpp)

# Include directories for external libraries, scoped to the target
target_include_directories(server PRIVATE 
    external/crow/include
    external/hnswlib
    external/json/single_include
    external/asio/asio/include
)

# Add platform-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(server PRIVATE 
        -pthread
        -march=native -mtune=native 
        -flto 
        -fvisibility=hidden 
        -funroll-loops 
        -Ofast 
        -DHAVE_CXX0X 
        -fopenmp 
        -fpic 
        -ftree-vectorize 
        -w
        -ffunction-sections 
        -fdata-sections
    )
endif()

# Link external libraries if needed
target_link_libraries(server PRIVATE pthread)

# Optional: Set output directory
set_target_properties(server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Optional: Make the build verbose for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)
