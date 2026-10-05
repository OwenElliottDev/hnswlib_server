cmake_minimum_required(VERSION 3.13)

# Browser/edge build of the index + datastore core. Configure with emcmake:
#   emcmake cmake -S wasm -B build-wasm && cmake --build build-wasm
# or just run wasm/build.sh.
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../VERSION HNSWLIB_VERSION LIMIT_COUNT 1)
project(HNSWLibEdge VERSION ${HNSWLIB_VERSION} LANGUAGES CXX)

if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "The wasm build must be configured with emcmake (Emscripten)")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ROOT}/VERSION)

add_executable(hnswlib_edge
    src/bindings.cpp
    src/edge_index.cpp
    ${ROOT}/src/data_store.cpp
    ${ROOT}/src/filters.cpp
)

target_include_directories(hnswlib_edge PRIVATE
    src
    ${ROOT}/src
    ${ROOT}/external/hnswlib
    ${ROOT}/external/json/single_include
)

# -msimd128 lets clang auto-vectorise the scalar distance loops with WASM SIMD.
# Native wasm exceptions are needed because the core reports errors by throwing.
set(EDGE_COMMON_FLAGS -fwasm-exceptions)
target_compile_options(hnswlib_edge PRIVATE ${EDGE_COMMON_FLAGS} -O3 -msimd128 -DHAVE_CXX0X)
target_compile_definitions(hnswlib_edge PRIVATE HNSWLIB_VERSION="${PROJECT_VERSION}")

target_link_options(hnswlib_edge PRIVATE
    ${EDGE_COMMON_FLAGS}
    -O3
    -lembind
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createHnswlibEdgeModule
    -sENVIRONMENT=web,worker,node
    -sALLOW_MEMORY_GROWTH=1
    -sMAXIMUM_MEMORY=4GB
    -sINITIAL_MEMORY=16MB
    -sSTACK_SIZE=1MB
    -sFORCE_FILESYSTEM=1
    -sEXPORTED_RUNTIME_METHODS=FS
    -sINCOMING_MODULE_JS_API=locateFile,wasmBinary,print,printErr
)

set_target_properties(hnswlib_edge PROPERTIES
    SUFFIX ".mjs"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dist
)
