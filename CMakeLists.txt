cmake_minimum_required(VERSION 3.20)
project(GaudiRT VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
    src/block_caching_allocator.cpp
    src/device_runtime.cpp
    src/pinned_host_pool.cpp
    src/recipe_registry.cpp
    src/sha256.cpp
)
target_link_libraries(gaudirt PUBLIC Synapse pthread ${CMAKE_DL_LIBS})

option(GAUDIRT_BUILD_TORCH "Build the generic PrivateUse1 extension" OFF)
if(GAUDIRT_BUILD_TORCH)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(Torch REQUIRED)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED NO_DEFAULT_PATH)
    add_library(_gaudirt_C MODULE
        src/bridge/generic_private_use1.cpp
        src/bridge/context.cpp
        src/bridge/context_allocator.cpp
        src/bridge/layout_copy.cpp
        src/bridge/paged_metadata.cpp
        src/bridge/attention_plan.cpp
        src/bridge/paged_attention.cpp
        src/bridge/paged_attention_tiles.cpp
        src/bridge/paged_kv_pool.cpp
        src/bridge/paged_slot_allocator.cpp
        src/bridge/eager_copy.cpp
        src/bridge/eager_ops.cpp
        src/bridge/eager_index.cpp
        src/bridge/eager_scheduler.cpp
        src/bridge/argmax_vocab.cpp
        src/bridge/bucket_program.cpp
        src/bridge/segmented_runner.cpp
        src/bridge/communicator.cpp
        src/bridge/process_group.cpp
        src/bridge/tensor_spec.cpp
        src/bridge/fx_node.cpp
        src/bridge/fx_geometry.cpp
        src/bridge/graph_builder.cpp
    )
    target_compile_definitions(_gaudirt_C PRIVATE TORCH_EXTENSION_NAME=_gaudirt_C)
    target_compile_options(_gaudirt_C PRIVATE ${TORCH_CXX_FLAGS})
    target_link_libraries(_gaudirt_C PRIVATE gaudirt hcl ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python3::Module)
    set_target_properties(_gaudirt_C PROPERTIES PREFIX "")

    add_executable(block_fp8_hybrid_lowering_unit_test
        src/bridge/bucket_program.cpp
        src/bridge/fx_geometry.cpp
        src/bridge/fx_node.cpp
        src/bridge/tensor_spec.cpp
        src/bridge/graph_builder.cpp
        src/block_caching_allocator.cpp
        src/sha256.cpp
        tests/block_fp8_hybrid_lowering_unit_test.cpp
    )
    target_include_directories(block_fp8_hybrid_lowering_unit_test BEFORE PRIVATE tests/mock_synapse src/bridge)
    target_compile_options(block_fp8_hybrid_lowering_unit_test PRIVATE ${TORCH_CXX_FLAGS} -UNDEBUG)
    target_link_libraries(block_fp8_hybrid_lowering_unit_test PRIVATE ${TORCH_LIBRARIES})

    add_executable(multitile_paged_attention_unit_test
        tests/multitile_paged_attention_unit_test.cpp
        src/bridge/attention_plan.cpp
        src/bridge/paged_attention.cpp
        src/bridge/paged_attention_tiles.cpp
        src/bridge/paged_metadata.cpp
        src/bridge/paged_kv_pool.cpp
        src/bridge/context.cpp
        src/bridge/layout_copy.cpp
        src/bridge/graph_builder.cpp
        src/block_caching_allocator.cpp
        src/device_runtime.cpp
        src/pinned_host_pool.cpp
        src/recipe_registry.cpp
        src/sha256.cpp
    )
    target_include_directories(multitile_paged_attention_unit_test BEFORE PRIVATE tests/mock_synapse src/bridge)
    target_compile_options(multitile_paged_attention_unit_test PRIVATE ${TORCH_CXX_FLAGS} -UNDEBUG)
    target_link_libraries(multitile_paged_attention_unit_test PRIVATE ${TORCH_LIBRARIES} pthread)
endif()

add_executable(gaudirt_runtime_unit_tests tests/runtime_unit_tests.cpp)
target_compile_options(gaudirt_runtime_unit_tests PRIVATE -UNDEBUG)
target_link_libraries(gaudirt_runtime_unit_tests PRIVATE gaudirt)

add_executable(block_allocator_fences_unit_test
    src/block_caching_allocator.cpp tests/block_allocator_fences_unit_test.cpp)
target_include_directories(block_allocator_fences_unit_test BEFORE PRIVATE tests/mock_synapse)
target_compile_options(block_allocator_fences_unit_test PRIVATE -UNDEBUG)

add_executable(host_policy_inherit_test tests/host_policy_inherit_test.cpp)
target_include_directories(host_policy_inherit_test PRIVATE include)

add_executable(recipe_cache_fingerprint_test tests/recipe_cache_fingerprint_test.cpp)
target_link_libraries(recipe_cache_fingerprint_test PRIVATE gaudirt)

add_executable(dynamic_pool_descriptor_device_test tests/dynamic_pool_descriptor_device_test.cpp)
target_link_libraries(dynamic_pool_descriptor_device_test PRIVATE gaudirt)

add_executable(fixed_kv_descriptor_device_test tests/fixed_kv_descriptor_device_test.cpp)
target_link_libraries(fixed_kv_descriptor_device_test PRIVATE gaudirt)

add_executable(fixed_tile_paged_attention_device_test tests/fixed_tile_paged_attention_device_test.cpp)
target_link_libraries(fixed_tile_paged_attention_device_test PRIVATE gaudirt)

add_executable(fused_paged_decode_device_test tests/fused_paged_decode_device_test.cpp)
target_link_libraries(fused_paged_decode_device_test PRIVATE gaudirt)

add_executable(attention_tile_indices_device_test tests/attention_tile_indices_device_test.cpp)
target_link_libraries(attention_tile_indices_device_test PRIVATE gaudirt)

add_executable(direct_softmax_device_test tests/direct_softmax_device_test.cpp)
target_link_libraries(direct_softmax_device_test PRIVATE gaudirt)

add_executable(block_fp8_device_test tests/block_fp8_device_test.cpp)
target_link_libraries(block_fp8_device_test PRIVATE gaudirt)

add_executable(compensated_qkv_reduce_device_test tests/compensated_qkv_reduce_device_test.cpp)
target_link_libraries(compensated_qkv_reduce_device_test PRIVATE gaudirt)

add_executable(stream_event_reuse_device_test tests/stream_event_reuse_device_test.cpp)
target_link_libraries(stream_event_reuse_device_test PRIVATE gaudirt)

add_executable(allocator_event_reuse_device_test tests/allocator_event_reuse_device_test.cpp)
target_link_libraries(allocator_event_reuse_device_test PRIVATE gaudirt)

add_executable(pinned_host_dma_device_test tests/pinned_host_dma_device_test.cpp)
target_link_libraries(pinned_host_dma_device_test PRIVATE gaudirt)

add_executable(pinned_host_fence_coalescing_device_test tests/pinned_host_fence_coalescing_device_test.cpp)
target_link_libraries(pinned_host_fence_coalescing_device_test PRIVATE gaudirt ${CMAKE_DL_LIBS})

add_executable(block_allocator_device_test tests/block_allocator_device_test.cpp)
target_link_libraries(block_allocator_device_test PRIVATE gaudirt)

add_library(gaudirt_synapse_call_counter SHARED tests/synapse_node_count_interposer.cpp)
target_link_libraries(gaudirt_synapse_call_counter PRIVATE ${CMAKE_DL_LIBS})
