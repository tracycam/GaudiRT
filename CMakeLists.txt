cmake_minimum_required(VERSION 3.20)
project(GaudiRT VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-O3 -Wall -Wextra -pthread)

set(HABANA_INC_DIR "/usr/include/habanalabs" CACHE PATH "Habana SDK include directory")
set(HABANA_LIB_DIR "/usr/lib/habanalabs" CACHE PATH "Habana SDK library directory")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${HABANA_INC_DIR}
)
link_directories(${HABANA_LIB_DIR})
add_link_options(-Wl,-rpath,${HABANA_LIB_DIR})

add_library(gaudirt STATIC
    src/device_runtime.cpp
    src/hbm_arena.cpp
    src/recipe_registry.cpp
    src/sha256.cpp
    src/weight_manifest.cpp
    src/tp_reduce.cpp
)
target_link_libraries(gaudirt PUBLIC Synapse hcl pthread)

add_executable(gaudirt_runtime_unit_tests tests/runtime_unit_tests.cpp)
target_compile_options(gaudirt_runtime_unit_tests PRIVATE -UNDEBUG)
target_link_libraries(gaudirt_runtime_unit_tests PRIVATE gaudirt)

add_library(mimo_layer_vertical STATIC
    models/mimo/mimo_attention.cpp
    models/mimo/mimo_full_attention.cpp
    models/mimo/mimo_dense_mlp.cpp
    models/mimo/mimo_moe.cpp
)
target_include_directories(mimo_layer_vertical PUBLIC models/mimo)
target_link_libraries(mimo_layer_vertical PUBLIC gaudirt)

add_executable(mimo_layer1 models/mimo/layer1_main.cpp)
target_link_libraries(mimo_layer1 PRIVATE mimo_layer_vertical)

add_executable(mimo_dense models/mimo/dense_main.cpp)
target_link_libraries(mimo_dense PRIVATE mimo_layer_vertical)

add_executable(mimo_full_attention models/mimo/full_attention_main.cpp)
target_link_libraries(mimo_full_attention PRIVATE mimo_layer_vertical)

add_library(mimo_layer0_vertical STATIC models/mimo/layer0.cpp)
target_include_directories(mimo_layer0_vertical PUBLIC models/mimo)
target_link_libraries(mimo_layer0_vertical PUBLIC mimo_layer_vertical)

add_executable(mimo_layer0 models/mimo/layer0_main.cpp)
target_link_libraries(mimo_layer0 PRIVATE mimo_layer0_vertical)

add_library(mimo_layer_b_vertical STATIC models/mimo/layer_b.cpp)
target_include_directories(mimo_layer_b_vertical PUBLIC models/mimo)
target_link_libraries(mimo_layer_b_vertical PUBLIC mimo_layer_vertical)

add_executable(mimo_layer_b models/mimo/layer_b_main.cpp)
target_link_libraries(mimo_layer_b PRIVATE mimo_layer_b_vertical)
