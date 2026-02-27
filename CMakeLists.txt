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

add_executable(server src/server.cpp src/data_store.cpp src/filters.cpp src/wal.cpp)

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

add_subdirectory(external/googletest)

# Test for filters.cpp
add_executable(test_filters tests/test_filters.cpp src/filters.cpp)
target_link_libraries(test_filters PRIVATE gtest gtest_main pthread)
target_include_directories(test_filters PRIVATE 
    external/crow/include
    external/hnswlib
    external/json/single_include
    external/asio/asio/include
    src
)

# Test for data_store.cpp
add_executable(test_data_store tests/test_data_store.cpp src/data_store.cpp src/filters.cpp) # Include filters.cpp here
target_link_libraries(test_data_store PRIVATE gtest gtest_main pthread)
target_include_directories(test_data_store PRIVATE 
    external/crow/include
    external/hnswlib
    external/json/single_include
    external/asio/asio/include
    src
)

# Test for test_datastore_stress.cpp
add_executable(test_datastore_stress tests/test_datastore_stress.cpp src/data_store.cpp src/filters.cpp) # Include filters.cpp here
target_link_libraries(test_datastore_stress PRIVATE gtest gtest_main pthread)
target_include_directories(test_datastore_stress PRIVATE 
    external/crow/include
    external/hnswlib
    external/json/single_include
    external/asio/asio/include
    src
)

# Test for save/load
add_executable(test_save_load tests/test_save_load.cpp src/data_store.cpp src/filters.cpp)
target_link_libraries(test_save_load PRIVATE gtest gtest_main pthread)
target_include_directories(test_save_load PRIVATE
    external/crow/include
    external/hnswlib
    external/json/single_include
    external/asio/asio/include
    src
)

# Enable testing
enable_testing()
add_test(NAME FiltersTest COMMAND test_filters)
add_test(NAME DataStoreTest COMMAND test_data_store)
add_test(NAME DataStoreStressTest COMMAND test_datastore_stress)
add_test(NAME SaveLoadTest COMMAND test_save_load)

# Test for WAL
add_executable(test_wal tests/test_wal.cpp src/wal.cpp src/data_store.cpp src/filters.cpp)
target_link_libraries(test_wal PRIVATE gtest gtest_main pthread)
target_include_directories(test_wal PRIVATE
    external/crow/include
    external/hnswlib
    external/json/single_include
    external/asio/asio/include
    src
)
add_test(NAME WalTest COMMAND test_wal)


option(ENABLE_PROFILING "Enable profiling with gperftools" OFF)

if(ENABLE_PROFILING)
    find_library(PROFILER_LIB profiler)
    if(PROFILER_LIB)
        target_link_libraries(test_datastore_stress PRIVATE ${PROFILER_LIB})
        message(STATUS "Profiling enabled for test_datastore_stress with gperftools")
    else()
        message(WARNING "gperftools not found. Profiling disabled.")
    endif()
endif()


set(CMAKE_VERBOSE_MAKEFILE ON)
