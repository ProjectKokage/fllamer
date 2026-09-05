cmake_minimum_required(VERSION 3.16)

project(llama_dart_bridge VERSION 0.2.4 LANGUAGES CXX)

set(LLAMA_DART_UPSTREAM_COMMIT ddd4ec1428a6201e18975ea52b07c71e0f9aef26)
set(LLAMA_DART_UPSTREAM_BUILD_NUMBER 10217)

option(
  LLAMA_DART_ENABLE_METAL
  "Build the Metal backend on Apple platforms"
  ${APPLE}
)
set(LLAMA_DART_ENABLE_VULKAN_DEFAULT OFF)
if(WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(LLAMA_DART_ENABLE_VULKAN_DEFAULT ON)
endif()
set(
  LLAMA_DART_ANDROID_VULKAN_HOST_ROOT
  ""
  CACHE PATH
  "Host Vulkan toolchain root for Android Vulkan builds"
)
set(
  LLAMA_DART_ANDROID_VULKAN_SHADER_OVERLAY_DIR
  ""
  CACHE PATH
  "Prepared Android Vulkan shader overlay directory"
)
option(
  LLAMA_DART_ENABLE_VULKAN
  "Require and build the Vulkan backend"
  ${LLAMA_DART_ENABLE_VULKAN_DEFAULT}
)
option(
  LLAMA_DART_NO_NETWORK
  "Disallow dependency downloads during native configuration and builds"
  ON
)
if(LLAMA_DART_ENABLE_METAL AND NOT APPLE)
  message(FATAL_ERROR "LLAMA_DART_ENABLE_METAL requires an Apple target")
endif()
set(LLAMA_DART_ANDROID_VULKAN_SAFE_QUANT 0)
set(LLAMA_DART_ANDROID_VULKAN_SAFE_K_QUANT 0)
if(ANDROID AND LLAMA_DART_ENABLE_VULKAN)
  if(NOT LLAMA_DART_ANDROID_VULKAN_HOST_ROOT)
    message(
      FATAL_ERROR
      "Android Vulkan builds require LLAMA_DART_ANDROID_VULKAN_HOST_ROOT"
    )
  endif()
  if(
    NOT LLAMA_DART_ANDROID_VULKAN_SHADER_OVERLAY_DIR OR
    NOT IS_DIRECTORY "${LLAMA_DART_ANDROID_VULKAN_SHADER_OVERLAY_DIR}"
  )
    message(
      FATAL_ERROR
      "Android Vulkan builds require the prepared shader overlay directory"
    )
  endif()
  if(NOT Vulkan_GLSLC_EXECUTABLE OR NOT EXISTS "${Vulkan_GLSLC_EXECUTABLE}")
    message(
      FATAL_ERROR
      "Android Vulkan builds require an explicit host Vulkan_GLSLC_EXECUTABLE"
    )
  endif()
  if(
    EXISTS
    "${LLAMA_DART_ANDROID_VULKAN_HOST_ROOT}/Include/vulkan/vulkan.hpp"
  )
    set(
      LLAMA_DART_ANDROID_VULKAN_HEADERS_INCLUDE_DIR
      "${LLAMA_DART_ANDROID_VULKAN_HOST_ROOT}/Include"
    )
  elseif(
    EXISTS
    "${LLAMA_DART_ANDROID_VULKAN_HOST_ROOT}/include/vulkan/vulkan.hpp"
  )
    set(
      LLAMA_DART_ANDROID_VULKAN_HEADERS_INCLUDE_DIR
      "${LLAMA_DART_ANDROID_VULKAN_HOST_ROOT}/include"
    )
  else()
    message(
      FATAL_ERROR
      "Android Vulkan header root does not contain vulkan/vulkan.hpp"
    )
  endif()
  set(
    LLAMA_DART_ANDROID_SPIRV_HEADERS_INCLUDE_DIR
    "${CMAKE_ANDROID_NDK}/sources/third_party/shaderc/third_party/spirv-tools/external/spirv-headers/include"
  )
  if(
    NOT EXISTS
    "${LLAMA_DART_ANDROID_SPIRV_HEADERS_INCLUDE_DIR}/spirv/unified1/spirv.hpp"
  )
    message(
      FATAL_ERROR
      "Android Vulkan builds require the SPIR-V headers bundled with the Android NDK"
    )
  endif()
  include_directories(
    SYSTEM
    "${LLAMA_DART_ANDROID_VULKAN_HEADERS_INCLUDE_DIR}"
    "${LLAMA_DART_ANDROID_SPIRV_HEADERS_INCLUDE_DIR}"
  )
  set(
    SPIRV-Headers_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/android-spirv-headers"
    CACHE PATH
    "Android NDK SPIR-V headers package configuration"
    FORCE
  )
  unset(Vulkan_ROOT CACHE)
  unset(Vulkan_ROOT)
  set(ENV{VULKAN_SDK} "")
  set(LLAMA_DART_ANDROID_VULKAN_SAFE_QUANT 1)
  set(LLAMA_DART_ANDROID_VULKAN_SAFE_K_QUANT 1)
elseif(LLAMA_DART_ANDROID_VULKAN_SHADER_OVERLAY_DIR)
  message(
    FATAL_ERROR
    "The Android Vulkan shader overlay is invalid outside an enabled Android Vulkan build"
  )
endif()
if(LLAMA_DART_NO_NETWORK)
  set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "" FORCE)
  set(LLAMA_LLGUIDANCE OFF CACHE BOOL "" FORCE)
elseif(NOT DEFINED LLAMA_LLGUIDANCE)
  set(LLAMA_LLGUIDANCE OFF CACHE BOOL "")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
set(LLAMA_TESTS_INSTALL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_HBM OFF CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
set(GGML_METAL ${LLAMA_DART_ENABLE_METAL} CACHE BOOL "" FORCE)
set(
  GGML_METAL_EMBED_LIBRARY
  ${LLAMA_DART_ENABLE_METAL}
  CACHE BOOL "" FORCE
)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_CANN OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_MUSA OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_OPENVINO OFF CACHE BOOL "" FORCE)
set(GGML_ET OFF CACHE BOOL "" FORCE)
set(GGML_RPC OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_VIRTGPU OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ${LLAMA_DART_ENABLE_VULKAN} CACHE BOOL "" FORCE)
set(GGML_WEBGPU OFF CACHE BOOL "" FORCE)
set(GGML_ZDNN OFF CACHE BOOL "" FORCE)
set(GGML_ZENDNN OFF CACHE BOOL "" FORCE)
set(GGML_HEXAGON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMIT ${LLAMA_DART_UPSTREAM_COMMIT})
set(LLAMA_BUILD_NUMBER ${LLAMA_DART_UPSTREAM_BUILD_NUMBER})

set(
  LLAMA_DART_BRIDGE_BUILD_FLAGS
  CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  LLAMA_DART_NO_NETWORK=${LLAMA_DART_NO_NETWORK}
  BUILD_SHARED_LIBS=OFF
  LLAMA_BUILD_COMMON=ON
  LLAMA_BUILD_TESTS=OFF
  LLAMA_BUILD_TOOLS=OFF
  LLAMA_BUILD_EXAMPLES=OFF
  LLAMA_BUILD_SERVER=OFF
  LLAMA_BUILD_APP=OFF
  LLAMA_BUILD_UI=OFF
  LLAMA_USE_PREBUILT_UI=OFF
  LLAMA_TOOLS_INSTALL=OFF
  LLAMA_TESTS_INSTALL=OFF
  LLAMA_OPENSSL=OFF
  LLAMA_LLGUIDANCE=${LLAMA_LLGUIDANCE}
  LLAMA_BUILD_MTMD=ON
  MTMD_VIDEO=OFF
  GGML_BACKEND_DL=OFF
  GGML_NATIVE=OFF
  GGML_OPENMP=OFF
  GGML_LLAMAFILE=OFF
  GGML_CPU_HBM=OFF
  GGML_CPU_KLEIDIAI=OFF
  GGML_METAL=${LLAMA_DART_ENABLE_METAL}
  GGML_METAL_EMBED_LIBRARY=${LLAMA_DART_ENABLE_METAL}
  GGML_BLAS=OFF
  GGML_ACCELERATE=OFF
  GGML_CANN=OFF
  GGML_CUDA=OFF
  GGML_HIP=OFF
  GGML_MUSA=OFF
  GGML_OPENCL=OFF
  GGML_OPENVINO=OFF
  GGML_ET=OFF
  GGML_RPC=OFF
  GGML_SYCL=OFF
  GGML_VIRTGPU=OFF
  GGML_VULKAN=${LLAMA_DART_ENABLE_VULKAN}
  GGML_VULKAN_ANDROID_SAFE_QUANT=${LLAMA_DART_ANDROID_VULKAN_SAFE_QUANT}
  GGML_VULKAN_ANDROID_SAFE_K_QUANT=${LLAMA_DART_ANDROID_VULKAN_SAFE_K_QUANT}
  GGML_WEBGPU=OFF
  GGML_ZDNN=OFF
  GGML_ZENDNN=OFF
  GGML_HEXAGON=OFF
)
string(JOIN " " LLAMA_DART_BRIDGE_BUILD_FLAGS_TEXT ${LLAMA_DART_BRIDGE_BUILD_FLAGS})

add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp
  EXCLUDE_FROM_ALL
)

if(ANDROID AND LLAMA_DART_ENABLE_VULKAN)
  include(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/android-vulkan-shader-overlay.cmake
  )
  fllamer_apply_android_vulkan_shader_overlay(
    ggml-vulkan
    "${LLAMA_DART_ANDROID_VULKAN_SHADER_OVERLAY_DIR}"
  )
endif()

add_library(
  llama_dart_bridge_reasoning STATIC
  src/reasoning_sampler.cpp
)
target_compile_features(llama_dart_bridge_reasoning PRIVATE cxx_std_17)
target_include_directories(
  llama_dart_bridge_reasoning PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(llama_dart_bridge_reasoning PUBLIC llama-common)

add_library(llama_dart_bridge SHARED src/llama_dart.cpp)
target_compile_features(llama_dart_bridge PRIVATE cxx_std_17)
target_compile_definitions(
  llama_dart_bridge PRIVATE
  LLAMA_DART_UPSTREAM_COMMIT="${LLAMA_DART_UPSTREAM_COMMIT}"
  LLAMA_DART_BUILD_FLAGS="${LLAMA_DART_BRIDGE_BUILD_FLAGS_TEXT}"
  LLAMA_DART_HAS_METAL=$<BOOL:${LLAMA_DART_ENABLE_METAL}>
  LLAMA_DART_HAS_VULKAN=$<BOOL:${LLAMA_DART_ENABLE_VULKAN}>
)
target_include_directories(
  llama_dart_bridge PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(
  llama_dart_bridge PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
target_link_libraries(
  llama_dart_bridge PRIVATE
  llama_dart_bridge_reasoning
  llama-common
  mtmd
)

if(APPLE)
  set(
    LLAMA_DART_EXPORT_CONTROL_FILE
    ${CMAKE_CURRENT_SOURCE_DIR}/exports_macos.txt
  )
  target_link_options(
    llama_dart_bridge PRIVATE
    "-Wl,-exported_symbols_list,${LLAMA_DART_EXPORT_CONTROL_FILE}"
  )
elseif(UNIX)
  set(
    LLAMA_DART_EXPORT_CONTROL_FILE
    ${CMAKE_CURRENT_SOURCE_DIR}/exports_unix.map
  )
  target_link_options(
    llama_dart_bridge PRIVATE
    "-Wl,--version-script=${LLAMA_DART_EXPORT_CONTROL_FILE}"
  )
endif()
if(LLAMA_DART_EXPORT_CONTROL_FILE)
  set_property(
    TARGET llama_dart_bridge APPEND PROPERTY
    LINK_DEPENDS ${LLAMA_DART_EXPORT_CONTROL_FILE}
  )
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(llama_dart_bridge_test tests/llama_dart_bridge_test.cpp)
  add_executable(
    llama_dart_reasoning_sampler_test
    tests/reasoning_sampler_test.cpp
  )
  if(MSVC)
    target_compile_options(llama_dart_bridge_test PRIVATE /UNDEBUG)
    target_compile_options(llama_dart_reasoning_sampler_test PRIVATE /UNDEBUG)
  else()
    target_compile_options(llama_dart_bridge_test PRIVATE -UNDEBUG)
    target_compile_options(llama_dart_reasoning_sampler_test PRIVATE -UNDEBUG)
  endif()
  target_include_directories(
    llama_dart_bridge_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(
    llama_dart_bridge_test PRIVATE llama_dart_bridge llama-common
  )
  target_link_libraries(
    llama_dart_reasoning_sampler_test PRIVATE
    llama_dart_bridge_reasoning
  )
  target_compile_definitions(
    llama_dart_bridge_test PRIVATE
    LLAMA_DART_TEST_VOCAB_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf"
    LLAMA_DART_TEST_QWEN_VOCAB_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp/models/ggml-vocab-qwen35.gguf"
    LLAMA_DART_TEST_TOOL_VOCAB_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp/models/ggml-vocab-gemma-4.gguf"
  )
  add_test(NAME llama_dart_bridge_test COMMAND llama_dart_bridge_test)
  add_test(
    NAME llama_dart_reasoning_sampler_test
    COMMAND llama_dart_reasoning_sampler_test
  )
  if(UNIX)
    add_test(
      NAME llama_dart_bridge_exports
      COMMAND
        ${CMAKE_COMMAND}
        -DNM=${CMAKE_NM}
        -DLIBRARY=$<TARGET_FILE:llama_dart_bridge>
        -DAPPLE_PLATFORM=${APPLE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_exports.cmake
    )
  endif()
endif()
