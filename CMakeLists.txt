cmake_minimum_required(VERSION 3.10)

project(HNSWLibServer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set build types
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# optimization flags for release
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffunction-sections -fdata-sections")

# static linking for non-Apple platforms
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++ -static -Wl,--gc-sections")
endif()

# enable LTO if supported
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
else()
    message(WARNING "LTO is not supported by the current compiler.")
endif()

add_executable(server src/server.cpp src/data_store.cpp src/filters.cpp)

target_include_directories(server PRIVATE 
    external/crow/include
    external/hnswlib
    external/json/single_include
    external/asio/asio/include
)

# platform-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(server PRIVATE 
        -pthread
        -march=native -mtune=native 
        -flto 
        -fvisibility=hidden 
        -funroll-loops 
        -Ofast 
        -DHAVE_CXX0X 
        -fpic 
        -ftree-vectorize 
        -w
        -ffunction-sections 
        -fdata-sections
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_options(server PRIVATE 
            -openmp
        )
       
    else()
        target_compile_options(server PRIVATE -fopenmp)
        target_link_libraries(server PRIVATE gomp pthread)
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(server PRIVATE 
        /O2 /Ob2 /Oi /Ot /GL /arch:AVX2
    )
endif()

# platform-specific linking
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(server PRIVATE ws2_32 mswsock)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # explicitly pthread pthread on macOS
    target_link_libraries(server PRIVATE pthread)
endif()

set_target_properties(server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set(CMAKE_VERBOSE_MAKEFILE ON)
