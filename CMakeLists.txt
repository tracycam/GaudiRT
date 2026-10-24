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
    src/execution_engine.cpp
)
target_link_libraries(gaudirt PUBLIC Synapse hcl pthread dl)

add_executable(prequeue_capacity_probe probes/prequeue_capacity_probe.cpp)
target_link_libraries(prequeue_capacity_probe PRIVATE gaudirt)

add_executable(g0_2_launch_and_sram_probe probes/g0_2_launch_and_sram_probe.cpp)
target_link_libraries(g0_2_launch_and_sram_probe PRIVATE gaudirt)

add_executable(g0_5_collective_handoff_probe probes/g0_5_collective_handoff_probe.cpp)
target_link_libraries(g0_5_collective_handoff_probe PRIVATE gaudirt)

add_executable(g0_7_distinct144_prequeue_probe probes/g0_7_distinct144_prequeue_probe.cpp)
target_link_libraries(g0_7_distinct144_prequeue_probe PRIVATE gaudirt)

add_executable(rt_1_2_3_runtime_gaps_probe probes/rt_1_2_3_runtime_gaps_probe.cpp)
target_link_libraries(rt_1_2_3_runtime_gaps_probe PRIVATE gaudirt)

add_library(scal_hook SHARED probes/p0_scal_hook.cpp)
target_link_libraries(scal_hook PUBLIC dl pthread)

add_executable(p0_scal_capture_decode_probe probes/p0_scal_capture_decode_probe.cpp)
target_link_libraries(p0_scal_capture_decode_probe PRIVATE scal_hook gaudirt)

add_executable(p1_autonomous_recipe_replay_probe probes/p1_autonomous_recipe_replay_probe.cpp)
target_link_options(p1_autonomous_recipe_replay_probe PRIVATE "LINKER:--no-as-needed")
target_link_libraries(p1_autonomous_recipe_replay_probe PRIVATE scal_hook gaudirt scal)

add_executable(p2a_autonomous_allgather_replay_probe probes/p2a_autonomous_allgather_replay_probe.cpp)
target_include_directories(p2a_autonomous_allgather_replay_probe PRIVATE /usr/include/drm)
target_link_libraries(p2a_autonomous_allgather_replay_probe PRIVATE gaudirt scal hl-thunk)

add_executable(p3a_compute_allgather_compute_probe probes/p3a_compute_allgather_compute_probe.cpp)
target_include_directories(p3a_compute_allgather_compute_probe PRIVATE /usr/include/drm)
target_link_options(p3a_compute_allgather_compute_probe PRIVATE "LINKER:--no-as-needed")
target_link_libraries(p3a_compute_allgather_compute_probe PRIVATE scal_hook gaudirt scal hl-thunk)

add_executable(gaudirt_runtime_unit_tests tests/runtime_unit_tests.cpp)
target_link_libraries(gaudirt_runtime_unit_tests PRIVATE gaudirt)
