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

add_executable(prequeue_capacity_probe probes/prequeue_capacity_probe.cpp)
target_link_libraries(prequeue_capacity_probe PRIVATE Synapse pthread dl)

add_executable(g0_2_launch_and_sram_probe probes/g0_2_launch_and_sram_probe.cpp)
target_link_libraries(g0_2_launch_and_sram_probe PRIVATE Synapse pthread dl)

add_executable(g0_5_collective_handoff_probe probes/g0_5_collective_handoff_probe.cpp)
target_link_libraries(g0_5_collective_handoff_probe PRIVATE Synapse hcl pthread dl)

add_executable(g0_7_distinct144_prequeue_probe probes/g0_7_distinct144_prequeue_probe.cpp)
target_link_libraries(g0_7_distinct144_prequeue_probe PRIVATE Synapse hcl pthread dl)
