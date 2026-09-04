add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        kvpp_mte_copy_def.cpp
    )
endif()

file(GLOB MEMFABRIC_HYBRID_INCLUDE_CANDIDATES
    LIST_DIRECTORIES true
    "/usr/local/memfabric_hybrid/*/include"
    "/usr/local/python*/lib/python*/site-packages/memfabric_hybrid/include"
    "/usr/local/lib/python*/site-packages/memfabric_hybrid/include"
)
find_path(MEMFABRIC_MTE_INCLUDE_DIR
    NAMES smem/device/smem_shm_aicore_base_api.h
    HINTS
        "$ENV{MEMFABRIC_HYBRID_HOME_PATH}/include"
        ${MEMFABRIC_HYBRID_INCLUDE_CANDIDATES}
)
if(NOT MEMFABRIC_MTE_INCLUDE_DIR)
    message(FATAL_ERROR
        "KvppMteCopy requires smem/device/smem_shm_aicore_base_api.h. "
        "Source the MemFabric Hybrid environment before building the op."
    )
endif()
include_directories(${MEMFABRIC_MTE_INCLUDE_DIR})

add_ops_compile_options(
    OP_NAME KvppMteCopy
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        --op_relocatable_kernel_binary=true
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE kvpp_mte_copy ACLNNTYPE aclnn_inner)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
