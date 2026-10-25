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
)
target_link_libraries(gaudirt PUBLIC Synapse hcl pthread)

add_executable(gaudirt_runtime_unit_tests tests/runtime_unit_tests.cpp)
target_link_libraries(gaudirt_runtime_unit_tests PRIVATE gaudirt)
