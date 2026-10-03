function(fllamer_apply_android_vulkan_shader_overlay target overlay_shader_dir)
  if(NOT TARGET "${target}")
    message(
      FATAL_ERROR
      "Pinned Android Vulkan target does not exist: ${target}"
    )
  endif()
  if(NOT TARGET vulkan-shaders-gen)
    message(FATAL_ERROR "Pinned vulkan-shaders-gen target does not exist")
  endif()
  if(NOT Vulkan_GLSLC_EXECUTABLE OR NOT EXISTS "${Vulkan_GLSLC_EXECUTABLE}")
    message(
      FATAL_ERROR
      "Android Vulkan shader overlay requires the selected host glslc"
    )
  endif()

  set(_overlay_dequant "${overlay_shader_dir}/dequant_funcs.glsl")
  set(_overlay_mul_mm "${overlay_shader_dir}/mul_mm_funcs.glsl")
  set(_overlay_mul_mm_comp "${overlay_shader_dir}/mul_mm.comp")
  set(_overlay_vulkan "${overlay_shader_dir}/ggml-vulkan.cpp")
  foreach(_overlay_file IN ITEMS
    "${_overlay_dequant}"
    "${_overlay_mul_mm}"
    "${_overlay_mul_mm_comp}"
    "${_overlay_vulkan}"
  )
    if(NOT EXISTS "${_overlay_file}")
      message(FATAL_ERROR "Android Vulkan shader overlay input is missing")
    endif()
  endforeach()
  set_property(
    DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_overlay_dequant}"
    "${_overlay_mul_mm}"
    "${_overlay_mul_mm_comp}"
    "${_overlay_vulkan}"
  )
  file(SHA256 "${_overlay_dequant}" _overlay_dequant_sha256)
  if(
    NOT _overlay_dequant_sha256 STREQUAL
    "bcaf084f68df8da15dcf2df3d15d45d6a443eccd9b53f26c78792e368079c7a8"
  )
    message(FATAL_ERROR "Android Vulkan dequant shader overlay is not exact")
  endif()
  file(SHA256 "${_overlay_mul_mm}" _overlay_mul_mm_sha256)
  if(
    NOT _overlay_mul_mm_sha256 STREQUAL
    "64ae33bf3fa95dfbed3fc17e2ed1e2e5405d3db193ad8a604b0e84f5536d665c"
  )
    message(FATAL_ERROR "Android Vulkan matrix shader overlay is not exact")
  endif()
  file(SHA256 "${_overlay_mul_mm_comp}" _overlay_mul_mm_comp_sha256)
  if(
    NOT _overlay_mul_mm_comp_sha256 STREQUAL
    "0a96c140a076875c6dc0e3fd0f29e5caceba7df8219da2b2067eb2cc1162af15"
  )
    message(FATAL_ERROR "Android Vulkan matrix shader views are not exact")
  endif()
  file(SHA256 "${_overlay_vulkan}" _overlay_vulkan_sha256)
  if(
    NOT _overlay_vulkan_sha256 STREQUAL
    "5c3d8a468f7cd742ff4a3ef513d648f350bfd3ee53dbce773895e0531714d3fc"
  )
    message(FATAL_ERROR "Android Vulkan native source overlay is not exact")
  endif()

  file(READ "${_overlay_dequant}" _overlay_dequant_text)
  string(
    FIND
    "${_overlay_dequant_text}"
    [=[const uint q0 = uint(data_a[a_offset + ib].qs[iqs    ]);]=]
    _q4_1_dequant_new
  )
  string(
    FIND
    "${_overlay_dequant_text}"
    [=[const i8vec2 v0 = unpack8(int32_t(data_a_packed16[a_offset + ib].qs[iqs/2])).xy;]=]
    _q8_0_dequant_old
  )
  string(
    FIND
    "${_overlay_dequant_text}"
    [=[int(data_a[a_offset + ib].qs[iqs + 3])]=]
    _q8_0_dequant_new
  )
  string(
    FIND
    "${_overlay_dequant_text}"
    [=[return (vec4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12) - 8.0f);]=]
    _q4_0_dequant_pristine
  )
  if(
    _q4_1_dequant_new EQUAL -1 OR
    NOT _q8_0_dequant_old EQUAL -1 OR
    _q8_0_dequant_new EQUAL -1 OR
    _q4_0_dequant_pristine EQUAL -1
  )
    message(FATAL_ERROR "Android Vulkan dequant overlay markers are invalid")
  endif()

  file(READ "${_overlay_mul_mm}" _overlay_mul_mm_text)
  string(
    FIND
    "${_overlay_mul_mm_text}"
    [=[const uint vui = uint(a_q4_0_p16.data[ib].qs[2*iqs])]=]
    _q4_0_matrix_old
  )
  string(
    FIND
    "${_overlay_mul_mm_text}"
    [=[const float d = float(a_q4_0_p16.data[ib].d);
        const uint qsi = 4 * iqs;
        const uvec4 q = uvec4(a_q4_0.data[ib].qs[qsi]=]
    _q4_0_matrix_new
  )
  string(
    FIND
    "${_overlay_mul_mm_text}"
    [=[const uint vui = a_q4_1_p32.data[ib].qs[iqs];
        const vec4 v0 = vec4(unpack8(vui & 0x0F0F0F0F)) * dm.x + dm.y;
        const vec4 v1 = vec4(unpack8((vui >> 4) & 0x0F0F0F0F)) * dm.x + dm.y;]=]
    _q4_1_matrix_old
  )
  string(
    FIND
    "${_overlay_mul_mm_text}"
    [=[const vec2 dm = vec2(a_q4_1_p32.data[ib].dm);
        const uint qsi = 4 * iqs;
        const uvec4 q = uvec4(a_q4_1.data[ib].qs[qsi]=]
    _q4_1_matrix_new
  )
  string(
    FIND
    "${_overlay_mul_mm_text}"
    [=[const i8vec2 v0 = unpack8(int32_t(a_q8_0_p16.data[ib].qs[2*iqs])).xy;]=]
    _q8_0_matrix_old
  )
  string(
    FIND
    "${_overlay_mul_mm_text}"
    [=[const vec4 v = vec4(int(a_q8_0.data[ib].qs[qsi    ])]=]
    _q8_0_matrix_new
  )
  if(
    NOT _q4_0_matrix_old EQUAL -1 OR
    _q4_0_matrix_new EQUAL -1 OR
    NOT _q4_1_matrix_old EQUAL -1 OR
    _q4_1_matrix_new EQUAL -1 OR
    NOT _q8_0_matrix_old EQUAL -1 OR
    _q8_0_matrix_new EQUAL -1
  )
    message(FATAL_ERROR "Android Vulkan matrix overlay markers are invalid")
  endif()

  file(READ "${_overlay_vulkan}" _overlay_vulkan_text)
  foreach(_overlay_vulkan_marker IN ITEMS
    [=[device->vendor_id == VK_VENDOR_ID_QUALCOMM &&]=]
    [=[device->driver_id == vk::DriverId::eQualcommProprietary;]=]
    [=[if (!ggml_vk_is_qualcomm_proprietary(device)) {]=]
    [=[src0->type == GGML_TYPE_Q4_K) {
        mmp_map = nullptr;
    }
    if (mmp_map == nullptr) {
        quantize_y = false;]=]
    [=[mul->src[0]->type == GGML_TYPE_Q4_K) {
            return false;]=]
    [=[src0_type == GGML_TYPE_Q5_K || src0_type == GGML_TYPE_Q6_K]=]
  )
    string(
      FIND
      "${_overlay_vulkan_text}"
      "${_overlay_vulkan_marker}"
      _overlay_vulkan_marker_position
    )
    if(_overlay_vulkan_marker_position EQUAL -1)
      message(FATAL_ERROR "Android Vulkan native overlay markers are invalid")
    endif()
  endforeach()

  get_target_property(_target_sources "${target}" SOURCES)
  get_target_property(_target_source_dir "${target}" SOURCE_DIR)
  if(NOT _target_sources OR NOT _target_source_dir)
    message(FATAL_ERROR "Pinned ggml-vulkan target shape is unavailable")
  endif()

  set(_normalized_sources "")
  foreach(_source IN LISTS _target_sources)
    if(NOT IS_ABSOLUTE "${_source}" AND NOT _source MATCHES "^\\$<")
      get_filename_component(
        _source
        "${_source}"
        ABSOLUTE
        BASE_DIR "${_target_source_dir}"
      )
    endif()
    list(APPEND _normalized_sources "${_source}")
  endforeach()

  set(_vulkan_source_matches "")
  foreach(_source IN LISTS _normalized_sources)
    get_filename_component(_source_name "${_source}" NAME)
    if(_source_name STREQUAL "ggml-vulkan.cpp")
      list(APPEND _vulkan_source_matches "${_source}")
    endif()
  endforeach()
  list(LENGTH _vulkan_source_matches _vulkan_source_count)
  if(NOT _vulkan_source_count EQUAL 1)
    message(FATAL_ERROR "Pinned ggml-vulkan.cpp target source shape changed")
  endif()
  list(GET _vulkan_source_matches 0 _vulkan_source)
  list(REMOVE_ITEM _normalized_sources "${_vulkan_source}")
  list(APPEND _normalized_sources "${_overlay_vulkan}")

  set(_shader_header_matches "")
  foreach(_source IN LISTS _normalized_sources)
    get_filename_component(_source_name "${_source}" NAME)
    if(_source_name STREQUAL "ggml-vulkan-shaders.hpp")
      list(APPEND _shader_header_matches "${_source}")
    endif()
  endforeach()
  list(LENGTH _shader_header_matches _shader_header_count)
  if(NOT _shader_header_count EQUAL 1)
    message(FATAL_ERROR "Pinned ggml-vulkan shader header shape changed")
  endif()
  list(GET _shader_header_matches 0 _shader_header)

  set(
    _overlay_binary_dir
    "${CMAKE_CURRENT_BINARY_DIR}/android-vulkan-shader-overlay"
  )
  set(_overlay_spv_dir "${_overlay_binary_dir}/spv")
  file(MAKE_DIRECTORY "${_overlay_binary_dir}" "${_overlay_spv_dir}")

  set(_host_executable_suffix "")
  if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(_host_executable_suffix ".exe")
  endif()
  set(
    _shader_generator
    "${CMAKE_BINARY_DIR}/$<CONFIG>/vulkan-shaders-gen${_host_executable_suffix}"
  )

  set(_overlay_outputs "")
  foreach(_shader IN ITEMS
    copy_from_quant.comp
    get_rows_quant.comp
    mul_mat_vec.comp
    mul_mm.comp
  )
    set(_original_matches "")
    foreach(_source IN LISTS _normalized_sources)
      get_filename_component(_source_name "${_source}" NAME)
      if(_source_name STREQUAL "${_shader}.cpp")
        list(APPEND _original_matches "${_source}")
      endif()
    endforeach()
    list(LENGTH _original_matches _original_count)
    if(NOT _original_count EQUAL 1)
      message(
        FATAL_ERROR
        "Pinned ${_shader}.cpp target source shape changed"
      )
    endif()
    list(GET _original_matches 0 _original_source)
    list(REMOVE_ITEM _normalized_sources "${_original_source}")

    set(_overlay_source "${overlay_shader_dir}/${_shader}")
    if(NOT EXISTS "${_overlay_source}")
      message(
        FATAL_ERROR
        "Android Vulkan shader overlay source is missing: ${_shader}"
      )
    endif()
    set(_overlay_cpp "${_overlay_binary_dir}/${_shader}.cpp")
    add_custom_command(
      OUTPUT "${_overlay_cpp}"
      DEPFILE "${_overlay_cpp}.d"
      COMMAND "${_shader_generator}"
        --glslc "${Vulkan_GLSLC_EXECUTABLE}"
        --source "${_overlay_source}"
        --output-dir "${_overlay_spv_dir}"
        --target-hpp "${_shader_header}"
        --target-cpp "${_overlay_cpp}"
      DEPENDS
        "${_overlay_source}"
        "${_overlay_dequant}"
        "${_overlay_mul_mm}"
        "${_overlay_mul_mm_comp}"
        vulkan-shaders-gen
      COMMENT "Generate fllamer Android Vulkan overlay for ${_shader}"
      VERBATIM
    )
    list(APPEND _normalized_sources "${_overlay_cpp}")
    list(APPEND _overlay_outputs "${_overlay_cpp}")
  endforeach()

  set_source_files_properties(
    ${_overlay_outputs}
    TARGET_DIRECTORY "${target}"
    PROPERTIES GENERATED TRUE
  )
  add_custom_target(
    fllamer-android-vulkan-shader-overlay
    DEPENDS ${_overlay_outputs}
  )
  add_dependencies("${target}" fllamer-android-vulkan-shader-overlay)
  set_property(TARGET "${target}" PROPERTY SOURCES "${_normalized_sources}")
  # The replaced source includes its sibling headers by quoted name, so the
  # directory it was taken from stays on the include path.
  get_filename_component(_vulkan_source_dir "${_vulkan_source}" DIRECTORY)
  target_include_directories("${target}" PRIVATE "${_vulkan_source_dir}")
endfunction()
